import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
from datetime import datetime
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

@st.cache_data(ttl=86400) # Caches for 24 hours
def fetch_nifty500_tickers():
    try:
        # Fetch directly from NSE official archives
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req) as response:
            df = pd.read_csv(response)
        
        # Ensure we get the Symbol column
        if 'Symbol' in df.columns:
            return [str(sym).strip() for sym in df['Symbol'].tolist() if pd.notna(sym)]
        else:
            raise KeyError("Symbol column not found")
    except Exception as e:
        st.sidebar.error("Failed to load Nifty 500 from NSE. Using fallback list.")
        return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "TATAMOTORS"] # Fallback

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

def analyze(df: pd.DataFrame, length: int, pct_lookback: int) -> dict:
    env = compute_envelope(df, length)
    if len(env) < length + 5: return {"status": "insufficient_data"}
    
    last_close = float(env["close"].iloc[-1])
    last_range = float(env["range"].iloc[-1])
    
    # Calculate historical percentile
    hist = env["range"].dropna().iloc[-pct_lookback:]
    pct_rank = float((hist < last_range).sum()) / len(hist) * 100 if len(hist) >= 10 else np.nan
    
    # NEW STRICT BUILD-UP CHECKS:
    # 1. Is price trapped inside the bands? (Hasn't broken out yet)
    upper_band = float(env["smooth"].iloc[-1])
    lower_band = float(env["smooth2"].iloc[-1])
    is_contained = bool(lower_band < last_close < upper_band)
    
    # 2. Is today's candle small/quiet? (Compare today's range to 14-day ATR)
    df["tr"] = df["High"] - df["Low"]
    atr_14 = float(df["tr"].ewm(span=14, adjust=False).mean().iloc[-1])
    today_candle_size = float(df["tr"].iloc[-1])
    is_quiet_today = bool(today_candle_size <= atr_14)
    
    wedge_len = max(1, length // 5)
    return {
        "status": "ok",
        "last_close": round(last_close, 2),
        "range_pct": round((last_range / last_close) * 100, 3) if last_close else None,
        "volatility_percentile": round(pct_rank, 1) if not np.isnan(pct_rank) else None,
        "contracting": bool(pine_falling(env["range"], length)),
        "wedge": bool(pine_rising(env["smooth2"], wedge_len) and pine_falling(env["smooth"], wedge_len)),
        "is_contained": is_contained,     # Added to payload
        "is_quiet_today": is_quiet_today, # Added to payload
        "as_of": str(env.index[-1]),
    }

# ==========================================
# 3. DATA FETCHING
# ==========================================
# ==========================================
# 3. DATA FETCHING
# ==========================================
def to_yahoo_symbol(symbol: str) -> str:
    return symbol if "." in symbol else f"{symbol}{EXCHANGE_SUFFIX}"

def fetch_ohlc(symbol: str, timeframe: str) -> pd.DataFrame:
    cfg = TIMEFRAME_MAP[timeframe]
    # Restored exact parameters from the Colab notebook to prevent formatting errors
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
                res = analyze(df, INDICATOR_LENGTH, PCT_LOOKBACK)
                if res.get("status") == "ok":
                    res.update({"symbol": sym, "timeframe": tf})
                    rows.append(res)
        except Exception as e:
            # Replaced the silent 'pass' so any future errors are printed to the screen
            st.toast(f"Skipped {sym} ({tf}): Data format error", icon="⚠️")
            
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return pd.DataFrame(rows)


# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("📉 CoilScan — Live Volatility Scanner")
st.markdown("Live scan across multiple timeframes for volatility contractions and squeezes.")

# Initialize Session State to hold data between UI clicks
if "scan_data" not in st.session_state:
    st.session_state.scan_data = pd.DataFrame()

# Sidebar Filters
with st.sidebar:
    st.header("⚙️ Scanner Settings")
    selected_tfs = st.multiselect("Timeframes", ALL_TIMEFRAMES, default=["1d"])
    
    if st.button("🔄 Run Live Scan", type="primary", use_container_width=True):
        with st.spinner(f"Scanning {len(WATCHLIST)} symbols..."):
            st.session_state.scan_data = run_scan(WATCHLIST, selected_tfs)
            st.session_state.last_run = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
    st.divider()
    st.header("🔍 Filter Results")
    search_q = st.text_input("Search Symbol (e.g. TATA)")
    pct_max = st.slider("Squeeze % max", 1, 50, 15)
    req_contracting = st.checkbox("Require contracting 🟢")
    req_wedge = st.checkbox("Require wedge (coil) 🟣")

# Main Display
df = st.session_state.scan_data

if df.empty:
    st.info("👈 Click **Run Live Scan** in the sidebar to fetch data.")
else:
    st.caption(f"Last scan: {st.session_state.get('last_run', 'N/A')}")
    
    # Apply UI filters to the DataFrame
    view = df.copy()
    if search_q:
        view = view[view["symbol"].str.contains(search_q, case=False)]
    
    # 1. Base filter: Must meet the maximum squeeze percentile
    view = view[view["volatility_percentile"].notna() & (view["volatility_percentile"] <= pct_max)]
    
    # 2. PREMIUM BUILD-UP FILTER: 
    # Must be trapped inside the bands AND today's candle must be quiet (no breakout yet)
    view = view[view["is_contained"] & view["is_quiet_today"]]
    
    # 3. User toggles
    if req_contracting: view = view[view["contracting"]]
    if req_wedge: view = view[view["wedge"]]
    
    if view.empty:
        st.warning("No stocks match the criteria. All tight setups have already broken out or failed the strict build-up check.")
    else:
        # Generate Signals
        def get_signal(row):
            if row.get("wedge"): return "🟣 Coil (wedge)"
            if row.get("contracting"): return "🟢 Contracting"
            return "🟡 Squeeze"
            
        view["Signal"] = view.apply(get_signal, axis=1)
        view = view.sort_values("volatility_percentile")
        view = view[["symbol", "timeframe", "last_close", "range_pct", "volatility_percentile", "Signal", "as_of"]]
        view.columns = ["Symbol", "Timeframe", "Last Close", "Envelope Width %", "Volatility Percentile", "Signal", "As Of"]
        
        # Color formatting
        def color_rows(row):
            if "Coil" in row["Signal"]: return ["background-color: rgba(108,142,245,0.15)"] * len(row)
            if "Contracting" in row["Signal"]: return ["background-color: rgba(53,196,136,0.12)"] * len(row)
            return ["background-color: rgba(232,163,61,0.12)"] * len(row)

        styled_df = view.style.apply(color_rows, axis=1).format({
            "Last Close": "{:.2f}", 
            "Envelope Width %": "{:.2f}%", 
            "Volatility Percentile": "{:.1f}"
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)
