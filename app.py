import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz

# ==============================================================================
# 1. PAGE CONFIGURATION & DARK THEME
# ==============================================================================
st.set_page_config(
    page_title="Quantum AI Binary Matrix",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Institutional Dark UI
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val { font-size: 26px; font-weight: bold; }
    .metric-title { font-size: 13px; color: #8b949e; text-transform: uppercase; }
    .alert-card-buy {
        background: linear-gradient(90deg, #134e4a 0%, #065f46 100%);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    .alert-card-sell {
        background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SIDEBAR - MANUAL SETTINGS & TUNING CONTROLS
# ==============================================================================
st.sidebar.title("⚙️ AI Control Center")

st.sidebar.subheader("1. AI Engine Thresholds")
ai_threshold = st.sidebar.slider("Minimum AI Conviction (%)", min_value=65, max_value=90, value=75, step=1,
                                 help="Higher conviction = Fewer, more accurate sniper signals.")
cooldown_bars = st.sidebar.slider("Signal Cooldown (Minutes)", min_value=1, max_value=5, value=2)

st.sidebar.subheader("2. Trading Session (Sri Lanka)")
use_session = st.sidebar.checkbox("Enable Sri Lanka Session Filter", value=True)
st.sidebar.caption("Sessions: 12:00-16:00 & 18:00-22:00 (Mon-Fri GMT+5:30)")

st.sidebar.subheader("3. Historical Chart Display")
history_days = st.sidebar.slider("History Depth (Days)", min_value=1, max_value=7, value=5)
enable_sound = st.sidebar.checkbox("Play Audio Alert Sound", value=True)

if st.sidebar.button("🔄 Refresh Market Data"):
    st.cache_data.clear()
    st.rerun()

# ==============================================================================
# 3. LIVE MARKET DATA INGESTION (EUR/USD 1M)
# ==============================================================================
@st.cache_data(ttl=30)
def load_market_data(days):
    ticker = yf.Ticker("EURUSD=X")
    df = ticker.history(period=f"{days}d", interval="1m")
    if df.empty:
        return pd.DataFrame()
    df.reset_index(inplace=True)
    # Standardize Column Names
    time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df.rename(columns={time_col: 'Time'}, inplace=True)
    # Convert to Sri Lanka Timezone (GMT+5:30)
    sl_tz = pytz.timezone('Asia/Colombo')
    df['Time'] = df['Time'].dt.tz_convert(sl_tz)
    return df

df = load_market_data(history_days)

if df.empty or len(df) < 50:
    st.error("Market data is currently unavailable. Please verify market hours or refresh.")
    st.stop()

# ==============================================================================
# 4. MULTI-INDICATOR QUANTITATIVE FEATURE PIPELINE
# ==============================================================================
close = df['Close'].values
open_p = df['Open'].values
high = df['High'].values
low = df['Low'].values
vol = df['Volume'].values
n = len(df)

# Vectorized ATR
tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
tr[0] = high[0] - low[0]
atr14 = pd.Series(tr).rolling(14).mean().values
atr_safe = np.maximum(atr14, 0.00005)

# EMAs & SMAs
ema9 = pd.Series(close).ewm(span=9, adjust=False).mean().values
ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
sma20 = pd.Series(close).rolling(20).mean().values

# RSI Multi-Periods (7, 14, 21)
def calc_rsi(series, period):
    delta = pd.Series(series).diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs)).values

rsi7 = calc_rsi(close, 7)
rsi14 = calc_rsi(close, 14)

# Stochastic %K & %D (14, 3)
low14 = pd.Series(low).rolling(14).min()
high14 = pd.Series(high).rolling(14).max()
stoch_k = (100 * (pd.Series(close) - low14) / (high14 - low14 + 1e-9)).rolling(3).mean().values

# MACD (12, 26, 9)
ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
macd_line = (ema12 - ema26).values
macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values

# Bollinger Bands (20, 2)
bb_std = pd.Series(close).rolling(20).std().values
bb_upper = sma20 + (2.0 * bb_std)
bb_lower = sma20 - (2.0 * bb_std)

# Candle Anatomy Physics
candle_range = high - low
candle_body = np.abs(close - open_p)
upper_wick = high - np.maximum(open_p, close)
lower_wick = np.minimum(open_p, close) - low

lower_wick_ratio = np.where(candle_range > 0, lower_wick / candle_range, 0)
upper_wick_ratio = np.where(candle_range > 0, upper_wick / candle_range, 0)

# ==============================================================================
# 5. MULTI-FACTOR AI ENSEMBLE SCORING MATRIX (0 - 100%)
# ==============================================================================
bull_scores = np.zeros(n)
bear_scores = np.zeros(n)

# Scoring Rules (Each confirmed factor adds statistical probability)
for i in range(20, n):
    b_score = 40.0
    s_score = 40.0

    # Trend Factors
    if close[i] > ema50[i]: b_score += 15.0
    if close[i] < ema50[i]: s_score += 15.0
    if ema9[i] > ema21[i]: b_score += 10.0
    if ema9[i] < ema21[i]: s_score += 10.0

    # Momentum Factors
    if rsi14[i] > 50 and rsi14[i] > rsi14[i-1]: b_score += 10.0
    if rsi14[i] < 50 and rsi14[i] < rsi14[i-1]: s_score += 10.0
    if rsi7[i] <= 30: b_score += 12.0
    if rsi7[i] >= 70: s_score += 12.0
    if macd_line[i] > macd_signal[i]: b_score += 8.0
    if macd_line[i] < macd_signal[i]: s_score += 8.0
    if stoch_k[i] <= 25: b_score += 10.0
    if stoch_k[i] >= 75: s_score += 10.0

    # Volatility Envelope Factors
    if low[i] <= bb_lower[i]: b_score += 12.0
    if high[i] >= bb_upper[i]: s_score += 12.0

    # Candle Absorption Physics
    if lower_wick_ratio[i] >= 0.35 and close[i] > open_p[i]: b_score += 15.0
    if upper_wick_ratio[i] >= 0.35 and close[i] < open_p[i]: s_score += 15.0

    # Normalization (Max 98%)
    bull_scores[i] = min(b_score, 98.0)
    bear_scores[i] = min(s_score, 98.0)

df['Bull_Score'] = bull_scores
df['Bear_Score'] = bear_scores

# ==============================================================================
# 6. SESSION GATEKEEPER & SIGNAL EXECUTION
# ==============================================================================
times = df['Time'].dt
weekday = times.weekday  # 0=Mon, 4=Fri
hours = times.hour

is_sess1 = (weekday < 5) & (hours >= 12) & (hours < 16)
is_sess2 = (weekday < 5) & (hours >= 18) & (hours < 22)
df['In_Session'] = (is_sess1 | is_sess2) if use_session else True

# Signal Triggers
signals = [None] * n
last_sig_idx = -100

for i in range(1, n - 1):
    if df['In_Session'].iloc[i] and (i - last_sig_idx >= cooldown_bars):
        b_sc = df['Bull_Score'].iloc[i]
        s_sc = df['Bear_Score'].iloc[i]

        if b_sc >= ai_threshold and b_sc > s_sc:
            signals[i] = 'BUY'
            last_sig_idx = i
        elif s_sc >= ai_threshold and s_sc > b_sc:
            signals[i] = 'SELL'
            last_sig_idx = i

df['Signal'] = signals

# ==============================================================================
# 7. BINARY WIN RATE CALCULATION (IMMEDIATE NEXT CANDLE EXCLUSIVELY)
# ==============================================================================
df['Next_Open'] = df['Open'].shift(-1)
df['Next_Close'] = df['Close'].shift(-1)

results = [None] * n
for i in range(n - 1):
    sig = df['Signal'].iloc[i]
    if sig == 'BUY':
        results[i] = 'WIN' if df['Next_Close'].iloc[i] > df['Next_Open'].iloc[i] else 'LOSS'
    elif sig == 'SELL':
        results[i] = 'WIN' if df['Next_Close'].iloc[i] < df['Next_Open'].iloc[i] else 'LOSS'

df['Result'] = results

# Statistical Calculations
trade_mask = df['Signal'].notna() & df['Result'].notna()
traded_df = df[trade_mask].copy()

total_trades = len(traded_df)
wins = len(traded_df[traded_df['Result'] == 'WIN'])
losses = len(traded_df[traded_df['Result'] == 'LOSS'])
overall_wr = (wins / total_trades * 100) if total_trades > 0 else 0.0

buy_trades = traded_df[traded_df['Signal'] == 'BUY']
buy_wins = len(buy_trades[buy_trades['Result'] == 'WIN'])
buy_wr = (buy_wins / len(buy_trades) * 100) if len(buy_trades) > 0 else 0.0

sell_trades = traded_df[traded_df['Signal'] == 'SELL']
sell_wins = len(sell_trades[sell_trades['Result'] == 'WIN'])
sell_wr = (sell_wins / len(sell_trades) * 100) if len(sell_trades) > 0 else 0.0

# ==============================================================================
# 8. DASHBOARD HEADER & REAL-TIME ALERTS
# ==============================================================================
st.title("⚡ Quantum AI Binary Matrix [EUR/USD 1M]")

# Check Latest Candle Signal
latest_candle = df.iloc[-1]
latest_sig = latest_candle['Signal']

if latest_sig == 'BUY':
    st.markdown(f"""
    <div class="alert-card-buy">
        <h2>🚨 ACTIVE SIGNAL: <strong>BUY (CALL)</strong> @ {latest_candle['Close']:.5f}</h2>
        <p style="font-size:18px;">AI Conviction: <strong>{latest_candle['Bull_Score']:.0f}%</strong> | 1-Minute Expiry Countdown Active</p>
    </div>
    """, unsafe_allow_html=True)
    if enable_sound:
        st.markdown('<audio autoplay src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"></audio>', unsafe_allow_html=True)
elif latest_sig == 'SELL':
    st.markdown(f"""
    <div class="alert-card-sell">
        <h2>🚨 ACTIVE SIGNAL: <strong>SELL (PUT)</strong> @ {latest_candle['Close']:.5f}</h2>
        <p style="font-size:18px;">AI Conviction: <strong>{latest_candle['Bear_Score']:.0f}%</strong> | 1-Minute Expiry Countdown Active</p>
    </div>
    """, unsafe_allow_html=True)
    if enable_sound:
        st.markdown('<audio autoplay src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"></audio>', unsafe_allow_html=True)

# Master Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown(f"<div class='metric-card'><div class='metric-title'>★ Overall Win Rate ★</div><div class='metric-val' style='color:#10b981;'>{overall_wr:.1f}%</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><div class='metric-title'>Total Signals</div><div class='metric-val'>{total_trades}</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><div class='metric-title'>Wins / Losses</div><div class='metric-val'><span style='color:#10b981;'>{wins}W</span> / <span style='color:#ef4444;'>{losses}L</span></div></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><div class='metric-title'>BUY Win Rate</div><div class='metric-val' style='color:#06b6d4;'>{buy_wr:.1f}%</div></div>", unsafe_allow_html=True)
col5.markdown(f"<div class='metric-card'><div class='metric-title'>SELL Win Rate</div><div class='metric-val' style='color:#f59e0b;'>{sell_wr:.1f}%</div></div>", unsafe_allow_html=True)

# ==============================================================================
# 9. INTERACTIVE CANDLESTICK CHART WITH HISTORICAL SIGNALS
# ==============================================================================
st.subheader("📈 Interactive 1M Candlestick Chart (Pinch & Scroll History)")

fig = go.Figure()

# Candlestick Series
fig.add_trace(go.Candlestick(
    x=df['Time'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="EUR/USD 1M",
    increasing_line_color="#10b981",
    decreasing_line_color="#ef4444"
))

# Mark BUY Signals directly on Chart
buy_pts = df[df['Signal'] == 'BUY']
if not buy_pts.empty:
    fig.add_trace(go.Scatter(
        x=buy_pts['Time'],
        y=buy_pts['Low'] - 0.00015,
        mode='markers+text',
        marker=dict(symbol='triangle-up', size=13, color='#10b981'),
        text=["BUY " + str(int(s)) + "%" for s in buy_pts['Bull_Score']],
        textposition="bottom center",
        name="BUY Signal"
    ))

# Mark SELL Signals directly on Chart
sell_pts = df[df['Signal'] == 'SELL']
if not sell_pts.empty:
    fig.add_trace(go.Scatter(
        x=sell_pts['Time'],
        y=sell_pts['High'] + 0.00015,
        mode='markers+text',
        marker=dict(symbol='triangle-down', size=13, color='#ef4444'),
        text=["SELL " + str(int(s)) + "%" for s in sell_pts['Bear_Score']],
        textposition="top center",
        name="SELL Signal"
    ))

fig.update_layout(
    template="plotly_dark",
    height=550,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor='#0d1117',
    plot_bgcolor='#0d1117'
)

st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 10. DETAILED HISTORICAL TRADE LOG (AUDIT TRAIL)
# ==============================================================================
st.subheader("📋 Historical Trade Verification Log")

if not traded_df.empty:
    log_df = traded_df[['Time', 'Signal', 'Close', 'Bull_Score', 'Bear_Score', 'Result']].copy()
    log_df['Score'] = np.where(log_df['Signal'] == 'BUY', log_df['Bull_Score'], log_df['Bear_Score']).astype(int).astype(str) + "%"
    log_df['Price'] = log_df['Close'].round(5)
    log_df['Time_Str'] = log_df['Time'].dt.strftime('%Y-%m-%d %H:%M')
    
    display_table = log_df[['Time_Str', 'Signal', 'Price', 'Score', 'Result']].iloc[::-1]
    
    def color_result(val):
        color = '#10b981' if val == 'WIN' else '#ef4444' if val == 'LOSS' else '#8b949e'
        return f'color: {color}; font-weight: bold;'

    st.dataframe(display_table.style.map(color_result, subset=['Result']), use_container_width=True, height=300)
else:
    st.info("No signals found in the current window with the selected AI conviction threshold. Lower the threshold slightly to see more trades.")
