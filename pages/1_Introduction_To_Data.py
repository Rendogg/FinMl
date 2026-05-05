import streamlit as st
import plotly.graph_objects as go
#Local
from utilFunctions import sidebar, get_data, verify_create_quiz, apply_theme


# Page Config
st.set_page_config(page_title="OHLCV Explorer", page_icon="🕯️", layout="wide")
apply_theme()

#Init Sidebar
sidebar("Introduction To Data")
data = get_data()

# UI
st.title("🕯️ OHLCV Data Explorer")
st.markdown("""
Before building any financial model, you need to understand the raw data it feeds on.
This lab walks you through **what OHLCV data is**, how each field relates to each other,
and how to visualise it using a **candlestick chart** , the standard for this data format.

""")
st.divider()

# What is OHLCV?
with st.expander("Step 1 , What is OHLCV Data?", expanded=False):
    st.markdown("""
    ### The building blocks of financial time series

    Every row in a financial dataset represents some **time period**: a minute, an hour, a day,
    a week etc. For each period, five values are recorded:

    | Field | Full Name | Definition |
    |-------|-----------|------------|
    | **O** | Open | Price at the **start** of the period |
    | **H** | High | The **highest** price reached during the period |
    | **L** | Low | The **lowest** price reached during the period |
    | **C** | Close | Price at the **end** of the period |
    | **V** | Volume | Total number of **shares traded** during the period |

    ### Why these five?

    Together they provide and overview about market activity:
    - **Open vs Close** → Did the price go **up** or **down** overall? By how much?
    - **High and Low** → How *volatile* was the period? How far did prices swing?
    - **Volume** → How much conviction was behind the move? High volume = strong signal.

    ### The flexibility of OHLCV

    The same format works at any time resolution. A "daily" bar summarises an entire trading
    day into five numbers. An "hourly" bar does the same for 60 minutes. This makes OHLCV
    universally applicable from high-frequency trading to long-term fundamental analysis.
    """)

    # Show a live sample of the actual data
    if st.checkbox("👀 Your data at a glance (Per Day)"):
        display_cols = ["Date", "Open", "High", "Low", "Close", "Volume"] if "Date" in data.columns \
            else ["Open", "High", "Low", "Close", "Volume"]
        st.dataframe(
            data[display_cols].tail(10).reset_index(drop=True),
            width='stretch'
        )

# How the fields relate
with st.expander("Step 2 , How the Fields Relate to Each Other", expanded=False):
    st.markdown("""
    ### Constraints and relationships

    OHLCV fields are closely related with meaningful insights to be discovered!

    ```
    Low  ≤  Open  ≤  High
    Low  ≤  Close ≤  High
    Low  ≤  High  (always)
    ```

    This means the High is *always* the largest value and the Low is *always* the smallest.
    The Open and Close sit somewhere in between.

    ### Expample derived metrics you'll may see

    | Derived Feature | Formula | What it captures |
    |----------------|---------|-----------------|
    | **Daily Return** | (Close − Prev Close) / Prev Close | % price change compared to previous day |
    | **True Range** | max(High−Low, abs(High−Prev Close), abs(Low−Prev Close)) | Volatility including gaps |
    | **Body size** | abs(Close − Open) | How decisive the price move was |
    | **Upper wick** | High − max(Open, Close) | Intraday rejection of higher prices |
    | **Lower wick** | min(Open, Close) − Low | Intraday rejection of lower prices |
    """)

    
    st.markdown("#### Summary statistics for your selected stock")
    stats = data[["Open", "High", "Low", "Close", "Volume"]].describe().round(2)
    st.dataframe(stats, width='stretch')

    violations = data[~((data["Low"] <= data["Open"]) &
                        (data["Open"] <= data["High"]) &
                        (data["Low"] <= data["Close"]) &
                        (data["Close"] <= data["High"]))]
    if len(violations) == 0:
        st.success(f"✅ All {len(data):,} rows satisfy Low ≤ Open/Close ≤ High")
    else:
        st.warning(f"⚠️ {len(violations)} rows violate OHLC constraints , possible data quality issues")
# Volume
with st.expander("Step 3 , The Role of Volume", expanded=False):
    st.markdown("""
    ### Volume: the market's conviction meter

    Price moves without volume are *weak signals*. A stock climbing on low volume might be
    a thin-market illusion. The same move on high volume suggests broad participation, buyers
    and sellers agree this is the right price.

    #### Key volume concepts

    | Concept | Meaning |
    |---------|---------|
    | **High volume + price up** | Strong bullish conviction |
    | **High volume + price down** | Strong selling pressure |
    | **Low volume + price up** | Weak rally, possible reversal |
    | **Low volume + price down** | Weak selloff, may not sustain |
    | **Volume spike** | Often precedes a major price move or news event |

    Volume is rarely used alone but is almost always analysed **alongside price**.
    Many technical indicators (OBV, VWAP, Money Flow Index) explicitly combine both.
    """)

    date_col = "Date" if "Date" in data.columns else data.index
    x_vals   = data["Date"] if "Date" in data.columns else data.index

    daily_return = data["Close"].pct_change()
    bar_colors   = ["#22c55e" if r >= 0 else "#ef4444" for r in daily_return.fillna(0)]

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=x_vals, y=data["Volume"],
        marker_color=bar_colors,
        name="Volume",
        opacity=0.8
    ))
    fig_vol.update_layout(
        template="plotly_dark",
        xaxis=dict(title="Date", type="date", tickformat="%b %Y"),
        yaxis_title="Shares Traded",
        height=300,
        margin=dict(t=10, b=10),
        showlegend=False,
        hovermode="x unified"
    )
    st.plotly_chart(fig_vol, width='stretch')
    st.caption("🟢 Green bars = price closed higher that day. 🔴 Red bars = price closed lower.")

# line chart vs candlestick
with st.expander("Step 4 , Why Not Just Use a Line Chart?", expanded=False):
    st.markdown("""
    ### The limitations with line charts

    A simple line chart of closing prices throws away **75% of your data**. It ignores
    Open, High, and Low entirely. You lose all information about *how* price moved within
    each period.

    Compare the two representations below:
    """)
    x_vals   = data["Date"] if "Date" in data.columns else data.index
    sample   = data.tail(60)
    x_sample = sample["Date"] if "Date" in data.columns else sample.index

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Line chart (Close only)**")
        fig_line = go.Figure(go.Scatter(
            x=x_sample, y=sample["Close"],
            mode="lines", line=dict(color="#00d4ff", width=2)
        ))
        fig_line.update_layout(
            template="plotly_dark", height=280,
            xaxis=dict(type="date", tickformat="%b %Y"),
            yaxis_title="Price", margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_line, width='stretch')

    with col_r:
        st.markdown("**Candlestick chart (O, H, L, C)**")
        fig_cs = go.Figure(go.Candlestick(
            x=x_sample,
            open=sample["Open"], high=sample["High"],
            low=sample["Low"],   close=sample["Close"],
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444"
        ))
        fig_cs.update_layout(
            template="plotly_dark", height=280,
            xaxis=dict(type="date", tickformat="%b %Y", rangeslider_visible=False),
            yaxis_title="Price", margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_cs, width='stretch')

    st.markdown("""
    The candlestick chart reveals *intraday overview*: was the period volatile? Did prices
    recover from a dip? Did a rally get rejected? None of that is visible in a line chart.
    """)

# Reading a candlestick
with st.expander("🕯️ Step 5 , Anatomy of a Candlestick", expanded=False):
    st.markdown("""
    ### How to read a single candle

    Each candlestick encodes four values in one compact visual:

    ```
            High ──── top of upper wick
              │
         ┌────┴────┐  ← top of body = max(Open, Close)
         │         │
         │  BODY   │  ← colour = direction (green = up, red = down)
         │         │
         └────┬────┘  ← bottom of body = min(Open, Close)
              │
            Low  ──── bottom of lower wick
    ```

    | Part | What it shows |
    |------|--------------|
    | **Body** | The range between Open and Close |
    | **Upper wick** | How high prices reached above the body before being rejected |
    | **Lower wick** | How low prices fell below the body before recovering |
    | **Body colour** | 🟢 Green = Close > Open (bullish). 🔴 Red = Close < Open (bearish) |

    ### Common single-candle patterns

    | Pattern | Shape | Meaning |
    |---------|-------|---------|
    | **Marubozu** | Large body, tiny/no wicks | Strong directional conviction |
    | **Doji** | Tiny body, long wicks | Indecision , Open ≈ Close |
    | **Hammer** | Small body, very long lower wick | Potential bullish reversal |
    | **Shooting Star** | Small body, very long upper wick | Potential bearish reversal |

    > Note: Single candle patterns are *context-dependent*, a hammer after a prolonged
    > downtrend is meaningful; the same shape mid-rally is not.
    """)

    # To Center the Image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            "https://blog.bettertrader.co/wp-content/uploads/2019/07/Candlesticks.png",
            caption="Bullish vs Bearish candlestick anatomy",
            width='stretch'
        )

# Build your own chart
st.subheader("Step 6 , Build Your Own Candlestick Chart")
st.markdown("""
Now it's your turn. Fill in the chart settings below to create a fully customised
candlestick chart of the stock you selected in the sidebar.
""")

with st.expander("ℹ️ Tips for good chart labels", expanded=False):
    st.markdown("""
    - **Title**: Be specific , include the ticker and time range (e.g. *"AAPL Daily Prices 2020–2024"*)
    - **X-axis**: Label the time resolution , *"Date"*, *"Trading Day"*, *"Month"*
    - **Y-axis**: Always include the unit , *"Price (USD)"*, *"Price (GBP)"*
    - **Colours**: Convention is green for bullish, red for bearish , but you can diverge for stylistic reasons
    """)

col_a, col_b = st.columns(2)
with col_a:
    chart_title = st.text_input("Chart title", placeholder="e.g. AAPL Daily Prices 2020–2024", key="chart_title")
    x_label     = st.text_input("X-axis label", placeholder="e.g. Date",          key="x_label")
with col_b:
    y_label     = st.text_input("Y-axis label", placeholder="e.g. Price (USD)",   key="y_label")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        bull_color = st.color_picker("Bullish colour 📈", value="#22c55e", key="bull_colour")
    with col_c2:
        bear_color = st.color_picker("Bearish colour 📉", value="#ef4444", key="bear_colour")

# Validation
missing = []
if not chart_title: 
    missing.append("Chart title")
if not x_label:
    missing.append("X-axis label")
if not y_label:
    missing.append("Y-axis label")
if bull_color == "#000000": 
    missing.append("Bullish colour (change from black)")
if bear_color == "#000000": 
    missing.append("Bearish colour (change from black)")

# Aggregate missing elements and display to the user
if missing:
    st.info(f"Please complete: **{', '.join(missing)}** to generate your chart.")
    st.stop()

st.success("✅ All settings complete , here's your chart!")

x_vals = data["Date"] if "Date" in data.columns else data.index
fig = go.Figure(go.Candlestick(
    x=x_vals,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    increasing_line_color=bull_color,
    decreasing_line_color=bear_color,
    name=chart_title
))

fig.update_layout(
    title=chart_title,
    xaxis=dict(
        title=x_label,
        type="date",
        tickformat="%b %Y",
        rangeslider_visible=False
    ),
    yaxis_title=y_label,
    template="plotly_dark",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, width='stretch')
st.divider()

# OHLCV with Volume overlay
st.subheader("📊 Candlestick + Volume , The Full Picture")

with st.expander("ℹ️ Why combine price and volume?", expanded=False):
    st.markdown("""
    Most professional charting tools show **volume as a bar chart directly beneath the
    candlestick chart**. This allows you to instantly correlate price movements with
    trading activity. Spotting high-conviction moves vs. thin-market noise at a glance.
    """)

fig_combo = go.Figure()

fig_combo.add_trace(go.Candlestick(
    x=x_vals,
    open=data["Open"], high=data["High"],
    low=data["Low"],   close=data["Close"],
    increasing_line_color=bull_color,
    decreasing_line_color=bear_color,
    name="Price",
    yaxis="y"
))

daily_return = data["Close"].pct_change()
vol_colors   = ["#22c55e" if r >= 0 else "#ef4444" for r in daily_return.fillna(0)]

fig_combo.add_trace(go.Bar(
    x=x_vals,
    y=data["Volume"],
    marker_color=vol_colors,
    opacity=0.9,
    name="Volume",
    yaxis="y2"
))

fig_combo.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    xaxis=dict(type="date", tickformat="%b %Y", rangeslider_visible=False),
    yaxis=dict(title=y_label or "Price", domain=[0.3, 1.0]),
    yaxis2=dict(title="Volume", domain=[0.0, 0.25], showgrid=False),
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
)

st.plotly_chart(fig_combo, width='stretch')

# Footer
st.divider()
st.info("""
**Lab Note , OHLCV as model input**  
Every machine learning model in this app : Linear Regression, Decision Trees, K-Means Clustering and Neural Networks ,
ultimately draws there features from this raw OHLCV data. Understanding what each field means
and how they relate is not just background knowledge: it directly informs which features
you engineer, which transformations make sense, and how to interpret your model's outputs.
""")

# Quiz
st.subheader("🧠 Knowledge Check , Test Your Understanding")

st.markdown("""
Answer the questions below to check your understanding of OHLCV data and candlestick charts.
Your results will be saved to your account.
""")

lab_specific_questions = [
    {
        "id": "q1",
        "type": "radio",
        "question": "What does the 'Close' price represent?",
        "options": ["High price", "Price at end of period", "Average"],
        "answer": "Price at end of period"
    },
     {
        "id": "q2",
        "type": "radio",
        "question": "Which condition must ALWAYS be true?",
        "options": ["Open ≥ High",
            "Low ≤ Open ≤ High",
            "Close ≥ High",
            "Volume ≥ High"],
        "answer": "Low ≤ Open ≤ High"
    },
      {
        "id": "q3",
        "type": "radio",
        "question": "What does high volume typically indicate?",
        "options": ["Weak price movement",
            "Low trader interest",
            "Strong market participation",
            "Price stability"],
        "answer": "Strong market participation"
    },
        {
        "id": "q4",
        "type": "radio",
        "question": " What does a long lower wick suggest?",
        "options": ["Prices were rejected at higher levels",
            "Prices dropped and recovered",
            "No movement occurred",
            "Strong upward trend only"],
        "answer": "Prices dropped and recovered"
    },
        {
        "id": "q5",
        "type": "radio",
        "question": "Why are candlestick charts better than line charts?",
        "options": ["They look nicer",
            "They only show closing prices",
            "They show Open, High, Low, Close information",
            "They remove volatility"],
        "answer": "They show Open, High, Low, Close information"
    },
    {
        "id": "q6", 
        "type": "text", 
        "question": "Explain Volume in your own words."
    }
]
verify_create_quiz(
    "OHLCV_Basics",
    lab_specific_questions
)
