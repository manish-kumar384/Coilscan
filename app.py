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
BATCH_SIZE = 40

TIMEFRAME_MAP = {
    "15m": {"interval": "15m", "period": "60d"},
    "30m": {"interval": "30m", "period": "60d"},
    "1h":  {"interval": "60m", "period": "730d"},
    # NSE 60m bars start at :15, so anchor 4h bins at 09:15 / 13:15
    "4h":  {"interval": "60m", "period": "730d", "resample": "4h", "offset": "1h15min"},
    "1d":  {"interval": "1d",  "period": "5y"},
    "1wk": {"interval": "1wk", "period": "10y"},
}
INTRADAY = {"15m", "30m", "1h", "4h"}

# (Base candles, Max base width %, Max distance to ceiling %)
TF_SETTINGS = {
    "15m": (10, 2.0, 0.5),
    "30m": (6,  2.0, 0.5),
    "1h":  (4,  3.0, 0.8),
    "4h":  (4,  4.0, 1.2),
    "1d":  (5,  6.0, 1.5),
    "1wk": (4,  8.0, 2.0),
}

FALLBACK = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "LT", "ITC", "AXISBANK", "BHARTIARTL"]


@st.cache_data(ttl=86400)
def fetch_nifty500_tickers():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            df = pd.read_csv(response)
        if "Symbol" in df.columns:
            syms = [str(s).strip() for s in df["Symbol"].tolist() if pd.notna(s) and "DUMMY" not in str(s)]
            if syms:
                return syms
    except Exception:
        pass
    return FALLBACK  # FIX: previously returned None if 'Symbol' column was missing


WATCHLIST = fetch_nifty500_tickers()

# ==========================================
# 2. INDICATOR LOGIC
# ==========================================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def pine_rising(series: pd.Series, length: int) -> bool:
    if len(series) < length + 1 or length < 1:
        return False
    return bool((series.diff().iloc[-length:] > 0).all())


def pine_falling(series: pd.Series, length: int) -> bool:
    if len(series) < length + 1 or length < 1:
        return False
    return bool((series.diff().iloc[-length:] < 0).all())


def compute_envelope(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    close = df["Close"].astype(float)
    basis = ema(close, length)
    d = ema((close - basis).abs(), length)
    upper, lower = basis + d, basis - d
    smooth = ema(pd.concat([upper, close], axis=1).max(axis=1), length)
    smooth2 = ema(pd.concat([close, lower], axis=1).min(axis=1), length)
    return pd.DataFrame(
        {"close": close, "range": smooth - smooth2, "smooth": smooth, "smooth2": smooth2},
        index=df.index,
    )


def analyze(df, length, pct_lookback, tf, min_touches=2, drop_last=False) -> dict:
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if drop_last:
        df = df.iloc[:-1]
    N, max_width, max_dist = TF_SETTINGS.get(tf, (5, 5.0, 1.5))
    if len(df) < max(length + 30, N + 25):
        return {"status": "insufficient_data"}

    env = compute_envelope(df, length)
    close = env["close"]
    last_close = float(close.iloc[-1])

    # FIX 1: normalise envelope width by price, so the percentile compares
    # like with like (raw price-width drifts as the stock trends).
    rng_pct = env["range"] / close * 100
    last_rp = float(rng_pct.iloc[-1])
    hist = rng_pct.iloc[-pct_lookback - 1:-1].dropna()
    pct_rank = float((hist < last_rp).mean() * 100) if len(hist) >= 30 else np.nan

    # FIX 2: "20 bars in a row strictly falling" almost never happens.
    # Use: narrower than k bars ago AND falling on most of the last k bars.
    k = max(3, length // 4)
    contracting = bool(
        last_rp < float(rng_pct.iloc[-1 - k]) and (rng_pct.diff().iloc[-k:] < 0).mean() >= 0.6
    )
    wedge_len = max(1, length // 5)
    wedge = bool(pine_rising(env["smooth2"], wedge_len) and pine_falling(env["smooth"], wedge_len))

    # FIX 3: ceiling/base is built from the PRIOR N bars (excluding the current one),
    # otherwise a breakout candle stretches its own base and always looks "at the ceiling".
    base = df.iloc[-N - 1:-1]
    ceiling = float(base["High"].max())
    floor = float(base["Low"].min())
    width_pct = (ceiling - floor) / floor * 100
    tight = width_pct <= max_width

    dist_pct = (ceiling - last_close) / ceiling * 100        # negative => closed above ceiling
    near = 0 <= dist_pct <= max_dist

    # FIX 4: a real flat top is TESTED repeatedly, not just "price is near the high".
    tol = ceiling * (1 - max_dist / 200)
    touches = int((base["High"] >= tol).sum())

    # Rising / flat lows = buyers stepping up under the ceiling (ascending base)
    half = max(1, N // 2)
    higher_lows = bool(base["Low"].iloc[half:].min() >= base["Low"].iloc[:half].min() * 0.998)

    uptrend = bool(last_close > float(ema(close, 50).iloc[-1]))

    # FIX 5: volume — dry-up inside the base, expansion on the breakout bar
    vol_dryup, vol_ratio = np.nan, np.nan
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        prior_vol = df["Volume"].iloc[-N - 21:-N - 1].mean()
        if prior_vol and prior_vol > 0:
            vol_dryup = float(base["Volume"].mean() / prior_vol)
        avg20 = df["Volume"].iloc[-21:-1].mean()
        if avg20 and avg20 > 0:
            vol_ratio = float(df["Volume"].iloc[-1] / avg20)

    # FIX 6: distinguish a pre-breakout setup from an actual breakout
    is_setup = bool(tight and near and touches >= min_touches and higher_lows)
    ext_pct = -dist_pct
    is_breakout = bool(tight and touches >= min_touches and 0 < ext_pct <= max_dist * 2)
    breakout_vol_ok = bool(is_breakout and not np.isnan(vol_ratio) and vol_ratio >= 1.5)

    pattern = "Breakout" if is_breakout else ("Setup" if is_setup else "")

    # FIX 7: rank results with a score instead of a single hard yes/no gate
    score = 0.0
    if pattern:
        score += (100 - pct_rank) * 0.30 if not np.isnan(pct_rank) else 0
        score += max(0, 1 - width_pct / max_width) * 20
        score += max(0, 1 - abs(dist_pct) / (max_dist * 2)) * 20
        score += min(touches, 4) / 4 * 10
        score += 10 if contracting else 0
        score += 5 if wedge else 0
        score += 5 if (not np.isnan(vol_dryup) and vol_dryup < 1) else 0
        score += 5 if uptrend else 0
        score += 5 if breakout_vol_ok else 0

    return {
        "status": "ok",
        "last_close": round(last_close, 2),
        "range_pct": round(last_rp, 3),
        "volatility_percentile": round(pct_rank, 1) if not np.isnan(pct_rank) else None,
        "contracting": contracting,
        "wedge": wedge,
        "pattern": pattern,
        "score": round(score, 1),
        "ceiling": round(ceiling, 2),
        "dist_pct": round(dist_pct, 2),
        "base_width_pct": round(width_pct, 2),
        "touches": touches,
        "higher_lows": higher_lows,
        "uptrend": uptrend,
        "vol_dryup": round(vol_dryup, 2) if not np.isnan(vol_dryup) else None,
        "vol_ratio": round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
        "base_n": N,
        "as_of": str(env.index[-1]),
    }


# ==========================================
# 3. DATA FETCHING (batched)
# ==========================================
def to_yahoo_symbol(symbol: str) -> str:
    return symbol if "." in symbol else f"{symbol}{EXCHANGE_SUFFIX}"


def _clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    if "resample" in cfg and not df.empty:
        df = (
            df.resample(cfg["resample"], offset=cfg.get("offset", "0min"))
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna(subset=["Close"])
        )
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_batch(symbols: tuple, timeframe: str) -> dict:
    """One Yahoo request for many symbols (much faster than 1 call per symbol)."""
    cfg = TIMEFRAME_MAP[timeframe]
    tickers = [to_yahoo_symbol(s) for s in symbols]
    raw = yf.download(
        tickers, interval=cfg["interval"], period=cfg["period"], auto_adjust=False,
        progress=False, group_by="ticker", threads=True,
    )
    out = {}
    if raw is None or raw.empty:
        return out
    for sym, tk in zip(symbols, tickers):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if tk not in raw.columns.get_level_values(0):
                    continue
                sub = raw[tk]
            else:
                sub = raw
            sub = sub.dropna(how="all")
            if sub.empty:
                continue
            out[sym] = _clean(sub, cfg)
        except Exception:
            continue
    return out


def fetch_ohlc(symbol: str, timeframe: str) -> pd.DataFrame:
    return fetch_batch((symbol,), timeframe).get(symbol, pd.DataFrame())


def run_scan(symbols, timeframes, min_touches, drop_last):
    rows, failed = [], 0
    chunks = [tuple(symbols[i:i + BATCH_SIZE]) for i in range(0, len(symbols), BATCH_SIZE)]
    total = len(chunks) * len(timeframes)
    progress_bar, done = st.progress(0), 0

    for tf in timeframes:
        for chunk in chunks:
            try:
                data = fetch_batch(chunk, tf)
            except Exception:
                data = {}
            failed += len(chunk) - len(data)
            for sym, df in data.items():
                try:
                    res = analyze(df, INDICATOR_LENGTH, PCT_LOOKBACK, tf, min_touches, drop_last)
                    if res.get("status") == "ok":
                        res.update({"symbol": sym, "timeframe": tf})
                        rows.append(res)
                except Exception:
                    failed += 1
            done += 1
            progress_bar.progress(done / total)

    progress_bar.empty()
    st.session_state.failed_count = failed
    return pd.DataFrame(rows)


# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("📉 CoilScan — Live Volatility Scanner")
st.markdown("Finds volatility squeezes with flat-top resistance build-ups, and fresh breakouts from them.")

if "scan_data" not in st.session_state:
    st.session_state.scan_data = pd.DataFrame()

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    selected_tfs = st.multiselect("Timeframes", ALL_TIMEFRAMES, default=["1d"])
    min_touches = st.slider("Min ceiling touches", 1, 4, 2)
    drop_last = st.checkbox("Ignore in-progress candle", value=False)

    if st.button("🔄 Run Live Scan", type="primary", width="stretch"):
        if not selected_tfs:
            st.warning("Select at least one timeframe.")
        else:
            with st.spinner(f"Scanning {len(WATCHLIST)} symbols..."):
                st.session_state.scan_data = run_scan(WATCHLIST, selected_tfs, min_touches, drop_last)
                st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.divider()
    st.header("🔍 Filter Results")
    search_q = st.text_input("Search Symbol (e.g. TATA)")
    patterns = st.multiselect("Pattern", ["Setup", "Breakout"], default=["Setup", "Breakout"])
    pct_max = st.slider("Volatility percentile max", 1, 100, 40)
    min_score = st.slider("Min score", 0, 100, 40)
    req_contracting = st.checkbox("Require contracting 🟢")
    req_wedge = st.checkbox("Require wedge (coil) 🟣")
    req_uptrend = st.checkbox("Require price above EMA50")
    req_vol = st.checkbox("Breakouts need volume ≥1.5x")

df = st.session_state.scan_data

if df.empty:
    st.info("👈 Click **Run Live Scan** in the sidebar to fetch data.")
else:
    st.caption(
        f"Last scan: {st.session_state.get('last_run', 'N/A')} · "
        f"{len(df)} analysed · {st.session_state.get('failed_count', 0)} failed/no data"
    )

    view = df.copy()
    if search_q:
        view = view[view["symbol"].str.contains(search_q, case=False)]
    view = view[view["pattern"].isin(patterns)]
    view = view[view["volatility_percentile"].notna() & (view["volatility_percentile"] <= pct_max)]
    view = view[view["score"] >= min_score]
    if req_contracting:
        view = view[view["contracting"]]
    if req_wedge:
        view = view[view["wedge"]]
    if req_uptrend:
        view = view[view["uptrend"]]
    if req_vol:
        view = view[(view["pattern"] != "Breakout") | (view["vol_ratio"].fillna(0) >= 1.5)]

    if view.empty:
        st.warning("No stocks match. Loosen the percentile / score filters or reduce ceiling touches.")
    else:
        def get_signal(row):
            if row["wedge"]:
                return "🟣 Coil (wedge)"
            if row["contracting"]:
                return "🟢 Contracting"
            return "🟡 Squeeze"

        view["Signal"] = view.apply(get_signal, axis=1)
        view["Pattern"] = view["pattern"].map({"Setup": "🎯 Setup", "Breakout": "🚀 Breakout"})
        view = view.sort_values("score", ascending=False)

        cols = ["symbol", "timeframe", "Pattern", "score", "last_close", "ceiling", "dist_pct",
                "base_width_pct", "touches", "volatility_percentile", "vol_dryup", "vol_ratio",
                "Signal", "as_of"]
        names = ["Symbol", "TF", "Pattern", "Score", "Close", "Ceiling", "Dist to Ceiling %",
                 "Base Width %", "Touches", "Vol Percentile", "Base Vol vs Prior", "Last Vol x Avg",
                 "Signal", "As Of"]
        display_view = view[cols].copy()
        display_view.columns = names
        st.dataframe(display_view, width="stretch", hide_index=True)

        # ==========================================
        # 5. TRADINGVIEW-STYLE CHART
        # ==========================================
        st.divider()
        st.subheader("📊 Cross-Verify Pattern")

        from lightweight_charts.widgets import StreamlitChart

        c1, c2 = st.columns(2)
        sym = c1.selectbox("Symbol", view["symbol"].unique())
        sym_rows = view[view["symbol"] == sym]
        tf_sel = c2.selectbox("Timeframe", sym_rows["timeframe"].unique())
        row = sym_rows[sym_rows["timeframe"] == tf_sel].iloc[0]

        with st.spinner("Loading chart..."):
            chart_df = fetch_ohlc(sym, tf_sel)
            if drop_last:
                chart_df = chart_df.iloc[:-1]
            chart_env = compute_envelope(chart_df, INDICATOR_LENGTH)

            tv_df = chart_df.reset_index()
            tv_df = tv_df.rename(columns={
                tv_df.columns[0]: "time",
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
            })
            # FIX: keep the time-of-day for intraday, otherwise all bars of a day collapse into one
            fmt = "%Y-%m-%d %H:%M:%S" if tf_sel in INTRADAY else "%Y-%m-%d"
            t = pd.to_datetime(tv_df["time"])
            if t.dt.tz is not None:
                t = t.dt.tz_localize(None)
            tv_df["time"] = t.dt.strftime(fmt)

            chart = StreamlitChart(width=900, height=550)
            chart.set(tv_df)

            up = chart.create_line(name="Upper Envelope", color="rgba(0,150,255,0.8)", width=2)
            up.set(pd.DataFrame({"time": tv_df["time"], "Upper Envelope": chart_env["smooth"].values}).dropna())

            lo = chart.create_line(name="Lower Envelope", color="rgba(0,150,255,0.8)", width=2)
            lo.set(pd.DataFrame({"time": tv_df["time"], "Lower Envelope": chart_env["smooth2"].values}).dropna())

            # Ceiling drawn over the same N+1 bars the scanner used (prior base + current bar)
            span = int(row["base_n"]) + 1
            if len(tv_df) >= span:
                res_line = chart.create_line(name="Resistance Ceiling", color="rgba(255,0,0,0.8)", width=2)
                res_line.set(pd.DataFrame({
                    "time": tv_df["time"].iloc[-span:],
                    "Resistance Ceiling": [float(row["ceiling"])] * span,
                }))

            chart.load()  # FIX: StreamlitChart must be loaded to render
