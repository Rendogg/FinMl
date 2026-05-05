import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utilFunctions import sidebar,get_data,verify_create_quiz,apply_theme

#Init SideBar
sidebar("Linear Regression Lab")
data = get_data()

ACTUAL_STYLE  = dict(color="rgba(239,71,111,0.6)", size=5, line=dict(width=1, color="white"))
LINE_BLUE     = dict(color="#118ab2", width=3)
LINE_ORANGE   = dict(color="#f4a261", width=2, dash="dot")
LAYOUT_BASE   = dict(template="plotly_white", hovermode="x unified",
                     legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                 xanchor="right", x=1))

def _base_layout(title: str) -> dict:
    return {**LAYOUT_BASE, "title": dict(text=title, x=0.5, xanchor="center",
                                         font=dict(size=22)),
            "xaxis_title": "Date", "yaxis_title": "Stock Price (USD)"}

apply_theme()
scaler = StandardScaler()

st.title("Linear Regression Lab")
st.markdown("""
This lab environment demonstrates how a **Linear Regression** model fits to data to make predictions of price values(Y axis)
using time stamps(X axis). The scatter plots represents the actual observed values while the line will visualise the 
trend over time. 
""")

st.markdown("""
## Raw Data
Before applying any model it's always worth visualising the raw dataset.
You can find trends, noise and outliers before anything is fitted.

- **Coloured dots** → Actual closing prices
- **Blue line** → Simple visual connection between points (toggle in legend)
""")


fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(x=data["Date"], y=data["Close"],
                                mode="markers", marker=ACTUAL_STYLE,
                                name="Closing Price"))
fig_raw.add_trace(go.Scatter(x=data["Date"], y=data["Close"],
                                mode="lines", line=LINE_BLUE,
                                name="Line Between Points", visible="legendonly"))
fig_raw.update_layout(**_base_layout("Stock Market Analysis: Closing Price"))
st.plotly_chart(fig_raw, width='stretch')

st.divider()

st.markdown("""
# 🧪 **The Linear Regression Model**

Linear Regression seeks to find a *line of best fit*. A line which minimizes its distance to every data point.  
It follows the linear equation:
""")

st.latex(r"\hat{y} = \beta_0 + \beta_1 x")

st.markdown("""
**Where:**
- **$\\beta_0$ (Intercept):** The value of $y$ when $x$ is 0  
- **$\\beta_1$ (Slope):** How much the price ($y$) is expected to change for every one-unit change in time ($x$)

In essence, Linear Regression tries to find the optimal values for these two coefficients (intercept and slope) that produce the best-fitting line.
""")

st.header("Train/Test Split Interactive Tool")
st.markdown("""
Data is partitioned **chronologically** so the model never sees future prices
during training. The 80/20 default is standard practice (Kohavi, 1995;
Hastie et al.).
""")

#data['Date_num'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
data['Date_num'] = np.arange(len(data))

split_size = st.slider("Train/Test Split %", 50, 90, 80) / 100

X_train, X_test, y_train, y_test = train_test_split(
    data[['Date_num']],
    data['Close'],
    test_size=1-split_size,
    shuffle=False
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

fig_lr = go.Figure()
fig_lr.add_trace(go.Scatter(x=data["Date"], y=data["Close"],
                            mode="markers", marker=ACTUAL_STYLE, name="Actual"))
fig_lr.add_trace(go.Scatter(x=data.loc[X_test.index, "Date"], y=y_pred,
                            mode="lines", line=LINE_BLUE, name="LR Prediction"))
fig_lr.update_layout(**_base_layout("Linear Regression Prediction vs Actual"))
st.plotly_chart(fig_lr, width='stretch')

#Metrics
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

st.subheader("Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="R² Score (Test)", 
        value=f"{test_r2:.3f}", 
        delta=f"{test_r2 - train_r2:.3f}",
        delta_color="normal",
        help="Percentage of variance(around the mean) explained by the model. A negative delta indicates the model performs worse on unseen data (overfitting)."
    )

with col2:
    st.metric(
        label="MAE", 
        value=f"${mae:.2f}",
        help="Mean Absolute Error: The average 'dollar amount' the prediction is off by. Unlike R², this gives you a tangible sense of error."
    )

with col3:
    st.metric(
        label="RMSE", 
        value=f"${rmse:.2f}",
        help="Root Mean Squared Error: Similar to MAE but penalises larger errors more heavily. Useful if you want to avoid big 'misses'."
    )

with col4:
    st.metric(
        label="MAPE", 
        value=f"{mape:.2f}%",
        help="Mean Absolute Percentage Error: The average error as a percentage of the actual price. Usually, <5% is considered good for stock data."
    )
    

    # RESIDUAL ANALYSIS
with st.expander("📉 Residual Analysis — is the model missing something?"):
    st.markdown("""
    A **residual** is the difference between what the model predicted and
    what actually happened:

    > **Residual = Actual − Predicted**

    If the model has captured all patterns in the data, residuals should
    look like **random noise** — no obvious shape or trend.

    If you see a clear pattern (a curve, a funnel, a wave) it means the
    model is *systematically wrong* in a way a better model could fix.
    This is exactly why we move to polynomial regression next.
    """)

    residuals = y_test.values - y_pred

    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=data.loc[X_test.index, "Date"], y=residuals,
        mode="markers+lines",
        marker=dict(color="rgba(244,162,97,0.7)", size=5),
        line=dict(color="rgba(244,162,97,0.3)", width=1),
        name="Residual"
    ))
    fig_resid.add_hline(y=0, line_dash="dash", line_color="white",
                        annotation_text="Zero residual (perfect prediction)")
    fig_resid.update_layout(
        **{**LAYOUT_BASE,
            "title": dict(text="Residuals over Time (Actual − Predicted)",
                            x=0.5, xanchor="center"),
            "xaxis_title": "Date",
            "yaxis_title": "Residual (USD)"}
    )
    st.plotly_chart(fig_resid, width='stretch')

    # Residual histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=residuals, nbinsx=40,
        marker_color="#118ab2", opacity=0.75, name="Residuals"
    ))
    fig_hist.update_layout(
        **{**LAYOUT_BASE,
            "title": dict(text="Residual Distribution", x=0.5, xanchor="center"),
            "xaxis_title": "Residual (USD)", "yaxis_title": "Count"}
    )
    st.plotly_chart(fig_hist, width='stretch')
    st.caption(
        "A bell-shaped, centred distribution suggests well-behaved errors. "
        "Heavy tails or skew suggest the model is consistently over- or under-predicting."
    )

st.divider()

# POLYNOMIAL REGRESSION
st.markdown("""
## Polynomial Regression

A straight line struggles with curved data. Polynomial regression adds
higher-power terms so the model can *bend*:

$$\\hat{y} = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\cdots + \\beta_n x^n$$

Higher degree → more flexible curve, but risk of **overfitting**.
""")

degree = st.slider("Polynomial Degree", 1, 10, 1)

poly         = PolynomialFeatures(degree=degree)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly  = poly.transform(X_test_scaled)


model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)

sorted_idx      = np.argsort(X_test["Date_num"].values)
X_test_sorted   = X_test.iloc[sorted_idx]
y_pred_sorted   = y_pred_poly[sorted_idx]

# ── EXTENSION 2: CONFIDENCE INTERVAL on polynomial ───────────────────────
# Approximate 95 % PI via residual std — simple but honest about uncertainty
residuals_train = y_train.values - model_poly.predict(X_train_poly)
residual_std    = np.std(residuals_train)
upper           = y_pred_sorted + 1.96 * residual_std
lower           = y_pred_sorted - 1.96 * residual_std

fig_poly = go.Figure()
fig_poly.add_trace(go.Scatter(x=data["Date"], y=data["Close"],
                                mode="markers", marker=ACTUAL_STYLE, name="Actual"))
# Shaded CI band
dates_test_sorted = data.loc[X_test_sorted.index, "Date"]
fig_poly.add_trace(go.Scatter(
    x=pd.concat([dates_test_sorted, dates_test_sorted[::-1]]),
    y=np.concatenate([upper, lower[::-1]]),
    fill="toself",
    fillcolor="rgba(17,138,178,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="95% Prediction Interval",
    hoverinfo="skip"
))
fig_poly.add_trace(go.Scatter(
    x=dates_test_sorted, y=y_pred_sorted,
    mode="lines", line=LINE_BLUE,
    name=f"Polynomial Prediction (degree={degree})"
))
fig_poly.update_layout(**_base_layout("Polynomial Regression Prediction vs Actual"))
st.plotly_chart(fig_poly, width='stretch')

st.markdown("""
**Reading the shaded band:**  
The blue band is an approximate **95% Prediction Interval** — the model
estimates the true price will fall within this range 95% of the time.
A wider band means the model is *less certain*. Notice how the band
grows towards the edges of the test window where the model has less
training data nearby.

> **Prediction Interval vs Confidence Interval:**  
> A confidence interval describes uncertainty about the *mean trend line*.  
> A prediction interval is wider — it accounts for the fact that individual
> prices scatter around that mean too. For trading decisions, the prediction
> interval is the more honest measure.
""")

#train_r2_poly = model_poly.score(X_train_poly, y_train)
#test_r2_poly  = model_poly.score(X_test_poly,  y_test)
#st.metric("R² (test)", f"{test_r2_poly:.2f}", delta=f"{test_r2_poly - train_r2_poly:.2f}")
#st.write("*(A negative delta suggests the model is overfitting the training data)*")



mae_poly = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
mape_poly = np.mean(np.abs((y_test - y_pred_poly) / y_test)) * 100
test_r2_poly = model_poly.score(X_test_poly, y_test)
train_r2_poly = model_poly.score(X_train_poly, y_train)

# --- Metric Display ---
st.subheader(f"Polynomial Performance (Degree {degree})")
p1, p2, p3, p4 = st.columns(4)

p1.metric(
    label="R² Score (Test)", 
    value=f"{test_r2_poly:.3f}", 
    delta=f"{test_r2_poly - train_r2_poly:.3f}",
    help="Higher is better. A large negative delta strongly suggests overfitting—the model is 'hallucinating' trends that don't exist in the test set."
)

p2.metric(
    label="Avg. Error (MAE)", 
    value=f"${mae_poly:.2f}",
    help="The average dollar amount the polynomial curve misses the actual price by."
)

p3.metric(
    label="RMSE", 
    value=f"${rmse_poly:.2f}",
    help="Root Mean Square Error: Penalises larger outliers. If this is much higher than MAE, the model is failing badly on specific days."
)

p4.metric(
    label="MAPE", 
    value=f"{mape_poly:.2f}%",
    help="Mean Absolute Percentage Error. For stock prices, anything under 5% is statistically impressive for a simple polynomial."
)

# --- Dynamic Warning ---
if degree > 5:
    st.warning(f"⚠️ **High Degree Alert:** A degree of {degree} often causes 'Runaway Polynomials' where the curve shoots to infinity outside the training range.")


# ── EXTENSION 3: OVERFITTING EXPLORER ──
with st.expander("🔬 Overfitting Explorer — degree vs R² across the range"):
    st.markdown("""
    The chart below trains a polynomial model at **every degree from 1 to 10**
    and records the R² on both training and test data.

    - **Train R²** will almost always rise with degree — the model gets
        better at memorising what it has already seen.
    - **Test R²** tells the real story. When it starts *falling* while
        train R² keeps climbing, the model has crossed into **overfitting**:
        it has learned the noise in the training data, not the signal.

    The degree where test R² peaks is usually your best choice.
    """)

    degrees  = list(range(1, 11))
    train_r2s, test_r2s = [], []

    for d in degrees:
        pf     = PolynomialFeatures(degree=d)
        Xtr    = pf.fit_transform(X_train)
        Xte    = pf.transform(X_test)
        m      = LinearRegression().fit(Xtr, y_train)
        train_r2s.append(m.score(Xtr, y_train))
        test_r2s.append(m.score(Xte,  y_test))

    fig_overfit = go.Figure()
    fig_overfit.add_trace(go.Scatter(
        x=degrees, y=train_r2s, mode="lines+markers",
        line=dict(color="#118ab2", width=2), name="Train R²"
    ))
    fig_overfit.add_trace(go.Scatter(
        x=degrees, y=test_r2s, mode="lines+markers",
        line=dict(color="#ef476f", width=2), name="Test R²"
    ))
    fig_overfit.add_vline(
        x=degree, line_dash="dash", line_color="rgba(255,255,255,0.4)",
        annotation_text=f"Your current degree ({degree})",
        annotation_position="top left"
    )
    fig_overfit.update_layout(
        **{**LAYOUT_BASE,
            "title": dict(text="Train vs Test R² by Polynomial Degree",
                            x=0.5, xanchor="center"),
            "xaxis_title": "Polynomial Degree",
            "yaxis_title": "R² Score"}
    )
    st.plotly_chart(fig_overfit, width='stretch')

    best_degree = int(np.argmax(test_r2s)) + 1
    st.info(f"📌 For this dataset and split, **degree {best_degree}** "
            f"achieves the highest test R² ({test_r2s[best_degree-1]:.3f}).")

st.divider()

# ── MOVING AVERAGE ──────────────────────
st.markdown("""
## 📊 Moving Average Strategy

A **moving average** smooths short-term noise by averaging a rolling window
of previous prices. Using two windows of different lengths as new *features*
gives the model a sense of both short- and long-term momentum. 

Smaller window → more sensitive, noisier signal  
Larger window → smoother, slower trend
""")

# USER-DEFINED WINDOW SIZES
col_w1, col_w2 = st.columns(2)
with col_w1:
    window_short = st.slider("Short-term window (days)", 2, 30,  3)
with col_w2:
    window_long  = st.slider("Long-term window (days)",  5, 200, 9)

if window_short >= window_long:
    st.warning("⚠️ Short-term window should be smaller than long-term window.")

data_ma = data.copy()
data_ma[f"S_{window_short}"] = data_ma["Close"].shift(1).rolling(window=window_short).mean()
data_ma[f"S_{window_long}"]  = data_ma["Close"].shift(1).rolling(window=window_long).mean()
data_ma.dropna(inplace=True)

feat_cols = [f"S_{window_short}", f"S_{window_long}"]
X_ma = data_ma[feat_cols]
y_ma = data_ma["Close"]

t = int(0.8 * len(data_ma))
X_ma_tr, X_ma_te = X_ma.iloc[:t], X_ma.iloc[t:]
y_ma_tr, y_ma_te = y_ma.iloc[:t], y_ma.iloc[t:]

model_ma = LinearRegression().fit(X_ma_tr, y_ma_tr)
y_ma_pred = pd.Series(model_ma.predict(X_ma_te),
                        index=y_ma_te.index, name="price")

fig_ma = go.Figure()

# 1. Actual Price (Full Range)
fig_ma.add_trace(go.Scatter(
    x=data_ma["Date"], y=data_ma["Close"],
    mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=1),
    name="Historical Price (Train)"
))

# 2. Actual Price (Test Only - highlighted)
fig_ma.add_trace(go.Scatter(
    x=data_ma["Date"].iloc[t:], y=y_ma_te,
    mode="lines", line=dict(color="#ef476f", width=2),
    name="Actual Price (Test)"
))

# 3. Predicted Price (Test Only)
fig_ma.add_trace(go.Scatter(
    x=data_ma["Date"].iloc[t:], y=y_ma_pred,
    mode="lines", line=dict(color="#118ab2", width=2, dash="dash"),
    name="Model Prediction"
))

# 4. Vertical Split Line
split_date = data_ma["Date"].iloc[t]
fig_ma.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="gray")
fig_ma.add_annotation(x=split_date, text="Start of Test Period", showarrow=False, yshift=10)

fig_ma.update_layout(**_base_layout("MA Strategy: Train vs Test Results"))
st.plotly_chart(fig_ma, width='stretch')

# Metrics
train_r2_ma = model_ma.score(X_ma_tr, y_ma_tr)
test_r2_ma  = model_ma.score(X_ma_te, y_ma_te)
mae_ma = mean_absolute_error(y_ma_te, y_ma_pred)
rmse_ma = np.sqrt(mean_squared_error(y_ma_te, y_ma_pred))
mape_ma = np.mean(np.abs((y_ma_te - y_ma_pred) / y_ma_te)) * 100

# Metrics Display
st.subheader(f"MA Strategy Performance ({window_short}d / {window_long}d)")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)

mcol1.metric(
    label="R² Score (Test)", 
    value=f"{test_r2_ma:.3f}", 
    delta=f"{test_r2_ma - train_r2_ma:.3f}",
    help="Measures how well the Moving Average features explain price movements. If the delta is deeply negative, the model found a pattern in the past that disappeared in the test set."
)

mcol2.metric(
    label="Avg. Error (MAE)", 
    value=f"${mae_ma:.2f}",
    help="On average, how many dollars the prediction deviates from the actual price."
)

mcol3.metric(
    label="RMSE", 
    value=f"${rmse_ma:.2f}",
    help="Root Mean Square Error. Higher than MAE if the model has a few 'major misses' during volatile days."
)

mcol4.metric(
    label="MAPE", 
    value=f"{mape_ma:.2f}%",
    help="Mean Absolute Percentage Error. Generally, under 2-3% is excellent for MA-based linear models."
)

# Feature Importance
with st.expander("⚖️ Feature Importance — which window does the model trust more?"):
    st.markdown(f"""
    The moving average model uses two features: the **{window_short}-day average**
    and the **{window_long}-day average**. After fitting, each feature receives
    a **coefficient** — how much the model adjusts its prediction per unit
    change in that feature.

    A larger absolute coefficient means the model leans more heavily on
    that window when deciding what price to predict.
    """)

    coef_df = pd.DataFrame({
        "Feature":     feat_cols,
        "Coefficient": model_ma.coef_,
        "Abs Weight":  np.abs(model_ma.coef_)
    }).sort_values("Abs Weight", ascending=True)

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        x=coef_df["Coefficient"],
        y=coef_df["Feature"],
        orientation="h",
        marker_color=["#ef476f" if c < 0 else "#118ab2"
                        for c in coef_df["Coefficient"]],
    ))
    fig_coef.update_layout(
        **{**LAYOUT_BASE,
            "title": dict(text="Feature Coefficients (Moving Average Model)",
                            x=0.5, xanchor="center"),
            "xaxis_title": "Coefficient Value",
            "yaxis_title": "Feature"}
    )
    st.plotly_chart(fig_coef, width='stretch')

    dominant = feat_cols[int(np.argmax(np.abs(model_ma.coef_)))]
    st.info(
        f"📌 The model weighted **{dominant}** more heavily "
        f"(coefficient: {model_ma.coef_[feat_cols.index(dominant)]:.4f}). "
        f"This suggests the {'short' if dominant == feat_cols[0] else 'long'}-term "
        f"trend was more predictive for this stock and period."
    )

    st.markdown(f"""
    **Intercept:** {model_ma.intercept_:.4f}  
    This is the baseline price the model predicts before either moving
    average has any influence — in practice it's usually small relative
    to the coefficients.
    """)

st.divider()

# Model Comparison
st.markdown("""
## 📋 Model Comparison

All three models trained on this page, evaluated side by side.
R² ranges from 0 (no better than the mean) to 1 (perfect predictions).
A negative value means the model is actively worse than guessing the mean.
""")

comparison = pd.DataFrame([
    {
        "Model":       "Linear Regression",
        "Train R²":    round(train_r2,   3),
        "Test R²":     round(test_r2,    3),
        "Overfit Gap": round(train_r2 - test_r2, 3),
    },
    {
        "Model":       f"Polynomial (degree={degree})",
        "Train R²":    round(train_r2_poly,  3),
        "Test R²":     round(test_r2_poly,   3),
        "Overfit Gap": round(train_r2_poly - test_r2_poly, 3),
    },
    {
        "Model":       f"Moving Average ({window_short}/{window_long})",
        "Train R²":    round(train_r2_ma,    3),
        "Test R²":     round(test_r2_ma,     3),
        "Overfit Gap": round(train_r2_ma - test_r2_ma,  3),
    },
])

# Visual bar chart comparison
fig_cmp = go.Figure()
fig_cmp.add_trace(go.Bar(
    name="Train R²", x=comparison["Model"], y=comparison["Train R²"],
    marker_color="#118ab2"
))
fig_cmp.add_trace(go.Bar(
    name="Test R²", x=comparison["Model"], y=comparison["Test R²"],
    marker_color="#ef476f"
))
fig_cmp.update_layout(
    **{**LAYOUT_BASE,
        "barmode": "group",
        "title": dict(text="Train vs Test R² — All Models",
                        x=0.5, xanchor="center"),
        "xaxis_title": "Model",
        "yaxis_title": "R² Score"}
)
st.plotly_chart(fig_cmp, width='stretch')

# Table with conditional colouring
def colour_overfit(val):
    if val > 0.1:
        return "color: #ef476f"   # red
    elif val < 0:
        return "color: #06d6a0"   # green 
    return ""

styled = (
    comparison.style
    .applymap(colour_overfit, subset=["Overfit Gap"])
    .format({"Train R²": "{:.3f}", "Test R²": "{:.3f}", "Overfit Gap": "{:.3f}"})
)
st.dataframe(styled, width='stretch', hide_index=True)

st.markdown("""
**Reading the Overfit Gap:**
- A large positive gap (red) means the model memorised training data but
    generalises poorly — be cautious about trusting its predictions.
- A gap near zero means training and test performance are consistent —
    the model has generalised well.
- A negative gap (green) means the model scored *higher* on unseen data,
    which is unusual and often just a favourable random split.
""")

# Quiz

st.subheader("🧠 Knowledge Check — Test Your Understanding")
st.markdown("""
Answer the questions below to check your understanding of Regression.
Your results will be saved to your account.
""")

lab_specific_questions = [
{
        "id": "q1",
        "type": "radio",
        "question": "In a simple linear regression model ($y = mx + b$), what does the 'm' (coefficient) represent in a financial context?",
        "options": ["The starting price of the asset", "The average rate of change in price per unit of time", "The total volume of shares traded","The error margin of the prediction"],
        "answer": "The average rate of change in price per unit of time"
},
{
    "id": "q2",
        "type": "radio",
        "question": "Why might you choose Polynomial Regression over Linear Regression for stock prices?",
        "options": ["It always guarantees a higher profit", "It can capture non-linear trends and curves in price action", 
                    "It simplifies the data into a straight line",
                    "It ignores outliers automatically"],
        "answer": "It can capture non-linear trends and curves in price action"
},
{
    "id": "q3",
        "type": "radio",
        "question": "What is the primary risk of using a very high-degree polynomial (e.g., degree 10) to fit 20 days of data?",
        "options": ["Underfitting: The model is too simple", 
                    "Multicollinearity: The features are related", 
                    "Overfitting: The model memorizes noise instead of the trend",
                    "The model will become a straight line"],
        "answer": "Overfitting: The model memorizes noise instead of the trend"
},
{
    "id": "q4",
        "type": "radio",
        "question": "Which metric is most commonly used to evaluate how well the regression line fits the data points?",
        "options": ["R-Squared ($R^2$)", 
                    "Silhouette Score", 
                    "Gini Impurity",
                    "The Elbow Method"],
        "answer": "R-Squared ($R^2$)"
},
{
        "id": "q6", 
        "type": "text", 
        "question": "From your understanding, how may early stock data influcence the models predictions."
    }
    
]
verify_create_quiz(
    "Linear Regression",
    lab_specific_questions
)

