import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
from datetime import datetime
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CoilScan Scanner", page_icon="📉", layout="wide")

# ==========================================
# 1. CONFIG & GLOBALS
# ==========================================
EXCHANGE_SUFFIX = ".NS"
ALL_TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d", "1wk"]
INDICATOR_LENGTH = 20
PCT_LOOKBACK = 100

TIMEFRAME_MAP = {
    "15m": {"interval": "15m", "period": "60d"},
    "30m": {"interval": "30m", "period": "60d"},
    "1h":  {"interval": "60m", "period": "730d"},
    "4h":  {"interval": "60m", "period": "730d", "resample": "4h"},
    "1d":  {"interval": "1d",  "period": "5y"},
    "1wk": {"interval": "1wk", "period": "10y"},
}

# Settings for the Flat-Top Build-up: (Candles Lookback, Max Base Width %, Max Distance to Ceiling %)
TF_SETTINGS = {
    "15m": (10, 2.0, 0.5),
    "30m": (6,  2.0, 0.5),
    "1h":  (4,  3.0, 0.8),
    "4h":  (4,  4.0, 1.2),
    "1d":  (5,  6.0, 1.5),
    "1wk": (4,  8.0, 2.0)
}

@st.cache_data(ttl=86400)
def fetch_nifty500_tickers():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            df = pd.read_csv(response)
        if 'Symbol' in df.columns:
            # FIX: Filter out 'DUMMY' and NaN symbols to prevent Yahoo Finance spam
            return [str(sym).strip() for sym in df['Symbol'].tolist() if pd.notna(sym) and 'DUMMY' not in str(sym)]
    except Exception:
        return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

WATCHLIST = fetch_nifty500_tickers()

# ==========================================
# 2. INDICATOR LOGIC
# ==========================================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def pine_rising(series: pd.Series, length: int) -> bool:
    if len(series) < length + 1 or length < 1: return False
    return bool((series.diff().iloc[-length:] > 0).all())

def pine_falling(series: pd.Series, length: int) -> bool:
    if len(series) < length + 1 or length < 1: return False
    return bool((series.diff().iloc[-length:] < 0).all())

def compute_envelope(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    close = df["Close"].astype(float)
    basis = ema(close, length)
    d = ema((close - basis).abs(), length)
    upper, lower = basis + d, basis - d
    smooth = ema(pd.concat([upper, close], axis=1).max(axis=1), length)
    smooth2 = ema(pd.concat([close, lower], axis=1).min(axis=1), length)
    return pd.DataFrame({"close": close, "range": smooth - smooth2, "smooth": smooth, "smooth2": smooth2}, index=df.index)

def analyze(df: pd.DataFrame, length: int, pct_lookback: int, tf: str) -> dict:
    env = compute_envelope(df, length)
    if len(env) < length + 5: return {"status": "insufficient_data"}
    
    last_close = float(env["close"].iloc[-1])
    last_range = float(env["range"].iloc[-1])
    
    hist = env["range"].dropna().iloc[-pct_lookback:]
    pct_rank = float((hist < last_range).sum()) / len(hist) * 100 if len(hist) >= 10 else np.nan
    
    N_bars, max_width_pct, max_dist_pct = TF_SETTINGS.get(tf, (5, 5.0, 1.5))
    
    if len(df) >= N_bars:
        recent_df = df.iloc[-N_bars:]
        period_high = float(recent_df["High"].max())
        period_low = float(recent_df["Low"].min())
        
        base_width_pct = ((period_high - period_low) / period_low) * 100
        is_tight_base = base_width_pct <= max_width_pct
        dist_to_ceiling_pct = ((period_high - last_close) / period_high) * 100
        is_pushing_resistance = dist_to_ceiling_pct <= max_dist_pct
        
        is_flat_top_buildup = bool(is_tight_base and is_pushing_resistance)
    else:
        is_flat_top_buildup = False

    wedge_len = max(1, length // 5)
    return {
        "status": "ok",
        "last_close": round(last_close, 2),
        "range_pct": round((last_range / last_close) * 100, 3) if last_close else None,
        "volatility_percentile": round(pct_rank, 1) if not np.isnan(pct_rank) else None,
        "contracting": bool(pine_falling(env["range"], length)),
        "wedge": bool(pine_rising(env["smooth2"], wedge_len) and pine_falling(env["smooth"], wedge_len)),
        "is_flat_top_buildup": is_flat_top_buildup,
        "as_of": str(env.index[-1]),
    }

# ==========================================
# 3. DATA FETCHING
# ==========================================
def to_yahoo_symbol(symbol: str) -> str:
    return symbol if "." in symbol else f"{symbol}{EXCHANGE_SUFFIX}"

@st.cache_data(ttl=300)
def fetch_ohlc(symbol: str, timeframe: str) -> pd.DataFrame:
    cfg = TIMEFRAME_MAP[timeframe]
    df = yf.download(
        to_yahoo_symbol(symbol), 
        interval=cfg["interval"], 
        period=cfg["period"], 
        auto_adjust=False, 
        progress=False, 
        multi_level_index=False
    )
    if df is None or df.empty: return pd.DataFrame()
    df = df.dropna(how="all")
    if "resample" in cfg:
        df = df.resample(cfg["resample"]).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(how="any")
    return df

def run_scan(symbols, timeframes):
    rows = []
    progress_bar = st.progress(0)
    total = len(symbols) * len(timeframes)
    
    for i, (tf, sym) in enumerate([(t, s) for t in timeframes for s in symbols]):
        try:
            df = fetch_ohlc(sym, tf)
            if not df.empty:
                res = analyze(df, INDICATOR_LENGTH, PCT_LOOKBACK, tf)
                if res.get("status") == "ok":
                    res.update({"symbol": sym, "timeframe": tf})
                    rows.append(res)
        except Exception:
            pass # Suppressed error handling for a cleaner UI
            
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return pd.DataFrame(rows)

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("📉 CoilScan — Live Volatility Scanner")
st.markdown("Live scan across multiple timeframes for volatility contractions and squeezes.")

if "scan_data" not in st.session_state:
    st.session_state.scan_data = pd.DataFrame()

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    selected_tfs = st.multiselect("Timeframes", ALL_TIMEFRAMES, default=["1d"])
    
    # FIX: Replaced use_container_width with width="stretch" to clear the deprecation warning
    if st.button("🔄 Run Live Scan", type="primary", width="stretch"):
        with st.spinner(f"Scanning {len(WATCHLIST)} symbols..."):
            st.session_state.scan_data = run_scan(WATCHLIST, selected_tfs)
            st.session_state.last_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
    st.divider()
    st.header("🔍 Filter Results")
    search_q = st.text_input("Search Symbol (e.g. TATA)")
    pct_max = st.slider("Squeeze % max", 1, 50, 15)
    req_contracting = st.checkbox("Require contracting 🟢")
    req_wedge = st.checkbox("Require wedge (coil) 🟣")

df = st.session_state.scan_data

if df.empty:
    st.info("👈 Click **Run Live Scan** in the sidebar to fetch data.")
else:
    st.caption(f"Last scan: {st.session_state.get('last_run', 'N/A')}")
    
    view = df.copy()
    if search_q:
        view = view[view["symbol"].str.contains(search_q, case=False)]
    
    view = view[view["volatility_percentile"].notna() & (view["volatility_percentile"] <= pct_max)]
    
    # The strict image-based flat-top filter
    view = view[view["is_flat_top_buildup"] == True]
    
    if req_contracting: view = view[view["contracting"]]
    if req_wedge: view = view[view["wedge"]]
    
    if view.empty:
        st.warning("No stocks match the criteria. Currently, no symbols are showing a tight flat-top resistance build-up.")
    else:
        def get_signal(row):
            if row.get("wedge"): return "🟣 Coil (wedge)"
            if row.get("contracting"): return "🟢 Contracting"
            return "🟡 Squeeze"
            
        view["Signal"] = view.apply(get_signal, axis=1)
        view = view.sort_values("volatility_percentile")
        display_cols = ["symbol", "timeframe", "last_close", "range_pct", "volatility_percentile", "Signal", "as_of"]
        display_view = view[display_cols].copy()
        display_view.columns = ["Symbol", "Timeframe", "Last Close", "Envelope Width %", "Volatility Percentile", "Signal", "As Of"]
        
        st.dataframe(display_view, width="stretch", hide_index=True)
        
        # ==========================================
        # 5. CHART VISUALIZATION TOOL
        # ==========================================
        st.divider()
        st.subheader("📊 Cross-Verify Pattern")
        
        # Create a dropdown from the filtered results
        symbol_options = view["symbol"].unique()
        selected_chart_sym = st.selectbox("Select a symbol to plot its resistance ceiling:", symbol_options)
        
        if selected_chart_sym:
            # Find the timeframe that triggered the signal for this symbol
            trigger_tf = view[view["symbol"] == selected_chart_sym]["timeframe"].iloc[0]
            
            with st.spinner("Loading chart data..."):
                chart_df = fetch_ohlc(selected_chart_sym, trigger_tf)
                chart_env = compute_envelope(chart_df, INDICATOR_LENGTH)
                
                # Plotly Chart
                fig = go.Figure()
                
                # 1. Candlesticks
                fig.add_trace(go.Candlestick(
                    x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], 
                    low=chart_df['Low'], close=chart_df['Close'], name='Price'
                ))
                
                # 2. Envelope Bands (The blue lines)
                fig.add_trace(go.Scatter(x=chart_env.index, y=chart_env['smooth'], line=dict(color='rgba(0,150,255,0.6)', width=1.5), name='Upper Envelope'))
                fig.add_trace(go.Scatter(x=chart_env.index, y=chart_env['smooth2'], line=dict(color='rgba(0,150,255,0.6)', width=1.5), name='Lower Envelope'))
                
                # 3. Flat-Top Resistance Ceiling (The horizontal blue line from your image)
                N_bars = TF_SETTINGS.get(trigger_tf)[0]
                if len(chart_df) >= N_bars:
                    recent_df = chart_df.iloc[-N_bars:]
                    period_high = recent_df["High"].max()
                    
                    fig.add_trace(go.Scatter(
                        x=[recent_df.index[0], recent_df.index[-1]], 
                        y=[period_high, period_high],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Resistance Ceiling'
                    ))
                
                fig.update_layout(
                    title=f"{selected_chart_sym} - {trigger_tf} Timeframe",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=550,
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                # Render the chart natively in the Streamlit app
                st.plotly_chart(fig, use_container_width=True)
