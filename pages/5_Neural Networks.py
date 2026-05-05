import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import math

from utilFunctions import sidebar
from utilFunctions import get_data,verify_create_quiz,apply_theme

# Setup
sidebar("Neural Networks")
get_data()
apply_theme()


# HELPERS
def create_sequences(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def build_lstm(lookback: int, layers: list[dict], learning_rate: float) -> Sequential:
    model = Sequential()
    for i, cfg in enumerate(layers):
        return_seq = i < len(layers) - 1
        if i == 0:
            model.add(LSTM(cfg["units"], return_sequences=return_seq, input_shape=(lookback, 1)))
        else:
            model.add(LSTM(cfg["units"], return_sequences=return_seq))
        if cfg["dropout"] > 0:
            model.add(Dropout(cfg["dropout"]))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

def plot_predictions(dates_train, y_train_actual, dates_test, y_test_actual, y_test_pred, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_train, y=y_train_actual, name="Training Data", line=dict(color="#4C9BE8", width=1.5)))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test_actual, name="Actual Price (Test)", line=dict(color="#A8DADC", width=2)))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test_pred, name="LSTM Prediction", line=dict(color="#F4845F", width=2, dash="dot")))
    fig.update_layout(
        title=f"LSTM Prediction vs Actual , {ticker}",
        xaxis_title="Date", yaxis_title="Close Price (USD)",
        template="plotly_dark", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig

def plot_loss(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history["loss"], name="Train Loss", line=dict(color="#4C9BE8", width=2)))
    if "val_loss" in history.history:
        fig.add_trace(go.Scatter(y=history.history["val_loss"], name="Val Loss", line=dict(color="#F4845F", width=2)))
    fig.update_layout(
        title="Training Loss per Epoch", xaxis_title="Epoch", yaxis_title="MSE Loss",
        template="plotly_dark", height=320
    )
    return fig

# LSTM Configuration

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🧠 LSTM Configuration")

    lookback = st.slider(
        "Lookback Window (days)", min_value=5, max_value=120, value=60, step=5,
        help="How many previous days the LSTM sees when predicting the next price."
    )
    test_split = st.slider(
        "Test Set Size (%)", min_value=10, max_value=40, value=20, step=5,
        help="Percentage of data held out for testing. The model never trains on this."
    )

    st.markdown("---")
    st.markdown("### Network Architecture")
    st.caption("Each block below is one LSTM layer + optional Dropout.")

    num_layers = st.number_input("Number of LSTM Layers", min_value=1, max_value=4, value=2, step=1)

    layer_configs = []
    for i in range(int(num_layers)):
        with st.expander(f"Layer {i+1}", expanded=(i == 0)):
            units = st.slider("Units (neurons)", 16, 256, 64, 16, key=f"units_{i}")
            dropout = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.05, key=f"dropout_{i}")
            layer_configs.append({"units": units, "dropout": dropout})

    st.markdown("---")
    st.markdown("### Training Settings")

    epochs = st.slider("Max Epochs", 10, 200, 50, 10)
    batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)
    learning_rate = st.select_slider(
        "Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001,
        format_func=lambda x: f"{x:.4f}"
    )
    early_stopping = st.checkbox("Early Stopping (patience = 10)", value=True)

    st.markdown("---")
    run_btn = st.button("▶️ Train Model", type="primary", width='stretch')

# UI

col1, col2, col3 = st.columns([0.5, 1, 0.5])
with col2:
    st.title("LSTM Neural Network")
    st.caption("Build and train a Long Short-Term Memory network on real stock price data.")

# WHAT IS A NEURAL NETWORK?
with st.expander("What is a Neural Network?", expanded=False):
    st.markdown("""
    ### The Brain Analogy

    A **neural network** is a machine learning model loosely inspired by how the human brain works.
    The brain is made up of billions of neurons, each connected to many others. When you learn
    something new, connections between neurons are strengthened. Neural networks work similarly:
    they are made up of layers of artificial *nodes* (neurons), connected by numerical *weights* that are
    adjusted during training.

    ### Layers: The Building Blocks
    Every neural network has at least three types of layers:

    | Layer | Role |
    |---|---|
    | **Input Layer** | Receives the raw data. In our case, a window of past stock prices. |
    | **Hidden Layer(s)** | Transforms the input through learned patterns. This is where the "intelligence" lives. |
    | **Output Layer** | Produces the final prediction , the next closing price. |

    ### How Does Learning Actually Work?

    Training a neural network is essentially an iterative correction process:

    1. **Forward Pass** , The network takes in data and produces a prediction.
    2. **Calculate Error** , The prediction is compared to the known correct answer. The difference is the *loss*.
    3. **Backpropagation** , The error is traced backwards through the network, and each weight is
       nudged slightly in the direction that would reduce the error.
    4. **Repeat** , This process repeats across thousands of examples across many *epochs*
       (full passes through the training data) until the error is minimised.

    > 💡 **Think of it like this:** imagine learning to throw darts. After each throw, you adjust
    > your aim slightly based on how far off you were. Over many throws, your aim improves.
    > Neural networks do the same thing, just with numbers instead of darts.

    ### Why Neural Networks for Stock Prices?

    Traditional rule-based models struggle with the non-linear, sequential nature of financial data.
    Neural networks, particularly the LSTM variant used here, are well suited to finding hidden
    patterns across time.
    """)



# WHAT MAKES LSTM SPECIAL?
with st.expander("What is an LSTM and Why Does It Handle Time Series?", expanded=False):
    st.markdown("""
    ### The Problem with Standard Neural Networks

    A standard neural network has no memory. When it processes today's stock price, it has
    completely forgotten about what happened yesterday. For something like image classification this is acceptable
    but for **time-series** data like stock prices, context means much more. A price spike
    means something very different if it follows a month-long uptrend compared to a sudden crash.

    ### Enter the Recurrent Neural Network (RNN)

    A **Recurrent Neural Network (RNN)** solves this by feeding its own output back into itself
    at each time step, creating a form of short-term memory. However, standard RNNs suffer from
    the **vanishing gradient problem** where over long sequences, the influence of early data points
    effectively disappears during training. The model "forgets" what happened more than a few
    steps back.

    ### How LSTM Solves This: The Memory Cell

    The **Long Short-Term Memory (LSTM)** network was specifically designed to overcome this limitation.
    Each LSTM unit contains a **cell state**. a timeline of information,
    alongside three  *gates* that control what information is kept, discarded, or passed forward.

    | Gate | What it does |
    |---|---|
    | 🚪 **Forget Gate** | Decides which information from the past should be erased |
    | 📥 **Input Gate** | Decides which new information from the current step should be stored |
    | 📤 **Output Gate** | Decides what part of the cell state is passed to the next step |

    Because these gates are *learned from data*, the LSTM figures out on its own what patterns
    are worth remembering over long periods. This makes it excellent at capturing long-term
    dependencies in stock data. Seasonal patterns, multi-week trends, or the lingering
    effect of a major news event.

    ### Why This Matters for Stock Prediction

    Stock prices are driven by momentum, trends, and memory of past events. An LSTM can learn:
    - That prices tend to recover after a dip of a certain magnitude
    - That a persistent uptrend over 30 days has predictive weight
    - That volatility clusters , high-volatility days tend to follow high-volatility days

    None of these patterns are reachable by a model without memory.
    """)



# DATA PIPELINE EXPLAINED

with st.expander("How Is the Data Prepared Before Training?", expanded=False):
    st.markdown("""
    ### Step 1: Extract Closing Prices

    Of all the columns in the stock dataset (Open, High, Low, Close, Volume), we use only the
    **daily closing price**. This is the most widely studied price point in financial modelling
    and represents the stocks final valuation for the day.

    ### Step 2: Normalise with Min-Max Scaling

    Neural networks are sensitive to the *scale* of their inputs. If prices range from $50 to $500,
    the large numbers can cause unstable gradients during training. We apply **Min-Max Scaling**,
    which compresses all values into the range [0, 1]:

    ```
    scaled = (price - min_price) / (max_price - min_price)
    ```

    After predictions are made, we reverse this transformation to get back to real dollar values.

    ### Step 3: Create Sliding Window Sequences

    An LSTM doesn't receive a single price we give it a **window** of consecutive prices.
    For a lookback of 60 days, each training example is:

    - **Input (X):** Days 1–60 of closing prices
    - **Target (y):** The price on Day 61

    The window then slides forward by one day, and the process repeats. This creates thousands of
    overlapping training examples from a single price series, which is essential for the model
    to learn temporal patterns.

    ### Step 4: Train / Test Split (Chronological)

    Unlike other machine learning models where data can be randomly shuffled, **time series data must
    never be shuffled**. Doing so would cause *data leakage* where the model is able to "see the future" during
    training. Instead, we split chronologically:

    - **Training set:** The earlier portion of the data. The model learns from this.
    - **Test set:** The most recent portion. The model has never seen this, so performance here
      reflects true generalisation.

    The test split percentage is configurable in the sidebar.
    """)

# PARAMETER GUIDE
with st.expander("Understanding Every Parameter , What Should I Set?", expanded=False):
    st.markdown("### Lookback Window (days)")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(f"**Your current setting:** {lookback} days")
    with col2:
        st.markdown("""
        The number of past trading days the LSTM uses to make each prediction.

        - **Too small (e.g. 5–10):** The model can only see very recent price movement, may misses
          medium-term trends. Good for capturing short-term momentum.
        - **Moderate (30–60):** Captures multi-week trends and is a solid default for most stocks.
        - **Large (90–120):** Captures longer seasonal patterns but requires more data and takes
          longer to train. May hurt performance if the relationship changes over time.

        **Recommendation:** Start with 30–60 days. Increase if your stock has clear seasonal patterns.
        """)

    st.markdown("---")
    st.markdown("### LSTM Units (Neurons per Layer)")
    col1, col2 = st.columns([1, 2])
    with col1:
        for i, cfg in enumerate(layer_configs):
            st.info(f"**Layer {i+1}:** {cfg['units']} units")
    with col2:
        st.markdown("""
        Each unit in an LSTM layer is a separate memory cell that independently learns to track
        a different pattern in the data.

        - **Few units (16–32):** A compact, fast model. May underfit and be too simple to capture
          complex patterns.
        - **Moderate (64–128):** A good balance. Enough capacity to learn meaningful patterns
          without excessive training time.
        - **Many units (192–256):** A large model with high capacity. Risk of overfitting if the
          dataset is small. Needs more training data and longer training time.

        **Recommendation:** Start with 64 units. If both train and val loss are high (underfitting),
        increase units. If val loss rises while train loss falls (overfitting), reduce units or
        increase dropout.
        """)

    st.markdown("---")
    st.markdown("### Dropout Rate")
    col1, col2 = st.columns([1, 2])
    with col1:
        for i, cfg in enumerate(layer_configs):
            st.info(f"**Layer {i+1}:** {cfg['dropout']:.0%} dropout")
    with col2:
        st.markdown("""
        Dropout layers are usually placed after hidden layers. During training, it **randomly deactivates a
        percentage of neurons** at each step. This prevents the network from relying too heavily
        on any single neuron, forcing it to learn more general patterns and avoid overfitting(Learning training data
        to well making it lese applicable to unseen data).

        - **0%:** Fine for small networks, but larger ones may overfit.
        - **10–20%:** A safe default.
        - **30–50%:** Use if you see clear overfitting in the loss curve.

        > ⚠️ Dropout is only applied during **training**. It is automatically disabled during
        > prediction, so it does not add noise to the model's final outputs.

        **Recommendation:** 0.2 (20%) is a solid starting point. Watch the val loss, if it
        diverges from train loss, its typical to increase dropout.
        """)

    st.markdown("---")
    st.markdown("### Number of LSTM Layers")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(f"**Your current setting:** {len(layer_configs)} layer(s)")
    with col2:
        st.markdown("""
        Stacking multiple LSTM layers allows the network to learn **hierarchical representations**
        of temporal patterns. Lower layers capture short-term fluctuations, while deeper layers
        capture broader, longer-range dependencies.

        - **1 layer:** Simple and fast. Often sufficient for shorter lookback windows and simpler
          price series.
        - **2 layers:** A common default. Adds depth without dramatically increasing training time.
        - **3–4 layers:** High capacity. Rarely necessary for financial time series and significantly
          increases the risk of overfitting.

        **Recommendation:** 2 layers is a strong default. Only add a third if 2-layer models
        consistently underperform and you have a large dataset.
        """)

    st.markdown("---")
    st.markdown("### Epochs")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(f"**Your current setting:** {epochs} max epochs")
    with col2:
        st.markdown("""
        One **epoch** is a complete pass through the entire training dataset. More epochs give
        the model more opportunities to learn, but too many can cause overfitting.

        - **Too few:** The model hasn't finished learning. Both train and val loss will still
          be falling when training stops.
        - **Well-chosen:** Loss curves have flattened and stabilised.
        - **Too many (without early stopping):** Val loss starts rising while train loss continues
          falling , a classic sign of overfitting.

        **Recommendation:** Enable Early Stopping. This automatically halts training when val
        loss stops improving, so the max epoch count just sets an upper bound. 50–100 epochs
        is a reasonable ceiling for most configurations.
        """)

    st.markdown("---")
    st.markdown("### Learning Rate")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(f"**Your current setting:** {learning_rate:.4f}")
    with col2:
        st.markdown("""
        The learning rate controls **how large a step** the model takes when adjusting its
        weights after each batch. It is one of the most important hyperparameters.

        - **Too high (e.g. 0.01):** The model takes large steps and may overshoot the optimal
          weights, causing the loss to oscillate or diverge rather than converge.
        - **Balanced (0.001):** The standard default for the Adam optimiser. Converges reliably
          on most problems.
        - **Too low (e.g. 0.0001):** The model makes tiny adjustments. Training is stable but
          very slow, and you may need many more epochs to reach a good solution.

        **Recommendation:** Start with 0.001. If training is unstable (loss spikes or oscillates),
        reduce to 0.0005 or 0.0001.
        """)

    st.markdown("---")
    st.markdown("### Batch Size")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(f"**Your current setting:** {batch_size} samples per batch")
    with col2:
        st.markdown("""
        During training, the dataset is split into **batches**. The model updates its weights
        after processing each batch, rather than after every individual example.

        - **Small batches (8–16):** Noisier gradient estimates, but can generalise better.
          Slower per epoch, but can converge with fewer epochs.
        - **Medium batches (32–64):** A good balance between speed and gradient quality.
          32 is the most commonly used default.
        - **Large batches (128):** Faster per epoch but can lead to sharper, less generalisable
          minima. May require a lower learning rate.

        **Recommendation:** 32 is a reliable starting point. Reduce to 16 if you see the model
        failing to generalise on small datasets.
        """)

    st.markdown("---")
    st.markdown("### Early Stopping")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(f"**Status:** {'Enabled ✓' if early_stopping else 'Disabled ✗'}")
    with col2:
        st.markdown("""
        Early Stopping monitors the **validation loss** after each epoch. If it fails to improve
        for 10 consecutive epochs (the *patience*), training is halted automatically and the
        model weights from the best epoch are restored.

        This serves two important purposes:
        1. **Prevents overfitting** , training stops before the model starts memorising
           rather than generalising.
        2. **Saves time** , no need to train for the full epoch count if convergence(model loss stabilises, Neuron weights have minimal change) is
           reached earlier.

        **Recommendation:** Always leave this enabled. The only reason to disable it is if you
        specifically want to observe what happens when training runs for the full epoch count.
        """)

# UNDERSTANDING THE RESULTS
with st.expander("How to Read the Results. Metrics and Charts Explained", expanded=False):
    st.markdown("""
    ### Performance Metrics

    After training, three metrics evaluate how accurate the model's predictions were on the
    **test set** , data the model has never seen during training.

    #### RMSE , Root Mean Squared Error
    ```
    RMSE = √( mean( (actual - predicted)² ) )
    ```
    The average prediction error **in dollar terms**, with larger errors penalised more heavily.
    An RMSE of `$5.00` means the model's predictions were on average $5 away from the actual price.
    Lower is better. Compare this against the stock's typical daily price range to judge significance.

    #### MAE , Mean Absolute Error
    ```
    MAE = mean( |actual - predicted| )
    ```
    The simple average prediction error in dollars, without squaring. Less sensitive to outliers
    than RMSE. If MAE is much lower than RMSE, a few large errors are
    dragging RMSE up.

    #### MAPE , Mean Absolute Percentage Error
    ```
    MAPE = mean( |actual - predicted| / actual ) × 100
    ```
    The average prediction error as a **percentage** of the actual price. This makes it easier
    to compare performance across stocks at very different price levels. A MAPE of 2% means
    predictions were off by 2% on average. Values below 5% are generally considered good.

    ---

    ### The Prediction Chart

    The chart shows three series:
    - **Blue , Training Data:** The historical prices the model learned from.
    - **Teal , Actual Price (Test):** The real prices in the test window. The model never saw these.
    - **Orange dashed , LSTM Prediction:** What the model predicted for the test window.

    Look for how closely the orange line tracks the teal line. A good model will capture the
    direction of price movements even if the exact values aren't perfect.

    > ⚠️ **Important caveat:** A model can appear to "follow" prices simply by predicting that
    > tomorrow's price will be close to today's (a naïve baseline). Strong performance should
    > be validated by checking whether the model correctly predicts *direction* of movement,
    > not just proximity.

    ---

    ### The Loss Curve

    The loss curve plots **MSE (Mean Squared Error) loss** across training epochs for both the
    training and validation sets.

    | Pattern | What it means |
    |---|---|
    | Both curves falling steadily | ✅ The model is learning well |
    | Both curves flat from the start | ⚠️ Learning rate may be too low, or the model is too simple |
    | Train loss falls, val loss rises | ❌ **Overfitting** , increase dropout or reduce units |
    | Loss oscillates wildly | ⚠️ Learning rate is too high , reduce it |
    | Val loss drops faster than train loss | Usually fine , the val set may be easier |

    Early Stopping will cut training when val loss stops improving, so in a well-configured
    model the curves should flatten rather than diverge.
    """)

# LIMITATIONS & RESPONSIBLE USE
with st.expander("Limitations & Responsible Use of This Model", expanded=False):
    st.markdown("""
    ### What This Model Does NOT Know

    The LSTM here is trained exclusively on **historical closing prices**. This means it has
    no knowledge of:

    - **Earnings reports, dividends, or company news**
    - **Macroeconomic indicators** (interest rates, inflation, GDP)
    - **Geopolitical events** (elections, trade policy, crises)
    - **Market sentiment** (fear indices, retail trading volume)
    - **Sector-wide movements** or correlated asset behaviour

    A price series alone is a limited signal. Events that cause sudden market moves are invisible
    to this model.

    ### Past Performance Does Not Guarantee Future Results

    The model is trained on historical patterns. If the stock's behaviour changes due to a
    structural shift (a new CEO, regulatory change, or industry disruption), the historical
    patterns the LSTM learned may become irrelevant. This is known as **distribution shift**
    and is one of the core challenges of financial ML.

    ### Overfitting Risk

    A model that performs very well on training data but poorly on the test set has *memorised*
    rather than *learned*. Always evaluate performance on the held-out test set, and treat training
    accuracy with scepticism.

    ### This Is an Educational Tool

    FinML Lab is designed to help you **understand** how machine learning can be applied to
    financial data. The outputs of this model should not be used as the sole basis for any
    real-world investment decision. All investing involves risk, and machine learning models
    are not a substitute for thorough financial research and professional advice.
    """)

# DATA CHECKS
ticker = st.session_state.get("ticker", "AAPL")
df_raw = st.session_state.get("data", None)

if df_raw is None or df_raw.empty:
    st.warning("No data loaded. Use the sidebar to select a ticker and reload data.")
    st.stop()

if "Date" in df_raw.columns:
    dates = pd.to_datetime(df_raw["Date"])
else:
    dates = pd.to_datetime(df_raw.index)

close = df_raw["Close"].values.reshape(-1, 1)

if len(close) < lookback + 20:
    st.error(
        f"Not enough data ({len(close)} rows) for a lookback of {lookback}. "
        "Extend your date range or reduce the lookback window."
    )
    st.stop()


# ARCHITECTURE SUMMARY
st.markdown("### Your Network Architecture")
arch_cols = st.columns(len(layer_configs) + 1)
for i, cfg in enumerate(layer_configs):
    with arch_cols[i]:
        st.metric(f"LSTM Layer {i+1}", f"{cfg['units']} units")
        st.caption(f"Dropout: {cfg['dropout']}")
with arch_cols[-1]:
    st.metric("Output Layer", "Dense(1)")
    st.caption("Predicts next close price")

st.markdown("---")

# TRAINING
if run_btn:

    with st.spinner("Preprocessing data…"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)

        split_idx = int(len(scaled) * (1 - test_split / 100))
        train_data = scaled[:split_idx]
        test_data  = scaled[split_idx - lookback:]

        X_train, y_train = create_sequences(train_data, lookback)
        X_test,  y_test  = create_sequences(test_data,  lookback)

        X_train = X_train.reshape(*X_train.shape, 1)
        X_test  = X_test.reshape(*X_test.shape, 1)

    model = build_lstm(lookback, layer_configs, learning_rate)

    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))

    progress_bar = st.progress(0, text="Training LSTM…")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    progress_bar.progress(100, text="Training complete ✓")

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred        = scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual      = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    dates_s     = dates.reset_index(drop=True)
    test_dates  = dates_s.iloc[split_idx:].values
    train_dates = dates_s.iloc[:split_idx].values
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

    rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
    mae  = mean_absolute_error(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

    # Results
    st.markdown("### 📈 Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE",  f"${rmse:.2f}",  help="Average prediction error in dollars. Lower is better.")
    m2.metric("MAE",   f"${mae:.2f}",   help="Mean absolute error , average distance from actual price.")
    m3.metric("MAPE",  f"{mape:.2f}%",  help="Error as a percentage of price. Below 5% is generally good.")
    m4.metric("Epochs trained", len(history.history["loss"]))

    # Metrics
    st.markdown("#### 🔍 What Do These Numbers Mean?")
    interp_col1, interp_col2, interp_col3 = st.columns(3)

    avg_price = float(np.mean(y_actual))
    with interp_col1:
        rmse_pct = (rmse / avg_price) * 100
        if rmse_pct < 3:
            st.success(f"**RMSE is {rmse_pct:.1f}% of avg price** , predictions are closely tracking actual values.")
        elif rmse_pct < 7:
            st.warning(f"**RMSE is {rmse_pct:.1f}% of avg price** , reasonable, but there's room for improvement.")
        else:
            st.error(f"**RMSE is {rmse_pct:.1f}% of avg price** , predictions are significantly off. Try adjusting the architecture.")

    with interp_col2:
        if mape < 3:
            st.success(f"**MAPE {mape:.1f}%** , strong predictive accuracy for financial time series.")
        elif mape < 7:
            st.warning(f"**MAPE {mape:.1f}%** , moderate accuracy. Consider adjusting lookback or units.")
        else:
            st.error(f"**MAPE {mape:.1f}%** , high error rate. Try increasing training data range or tweaking parameters.")

    with interp_col3:
        epochs_used = len(history.history["loss"])
        if early_stopping and epochs_used < epochs:
            st.info(f"**Early Stopping triggered** at epoch {epochs_used}/{epochs}. The model converged before the limit , weights from the best epoch were restored.")
        else:
            st.info(f"**Trained for full {epochs_used} epochs.** Consider enabling Early Stopping to prevent overfitting.")

    st.plotly_chart(
        plot_predictions(train_dates, y_train_inv, test_dates, y_actual, y_pred, ticker),
        width='stretch'
    )

    col_loss, col_info = st.columns([2, 1])
    with col_loss:
        st.plotly_chart(plot_loss(history), width='stretch')
    with col_info:
        st.markdown("#### Reading the Loss Curve")

        train_loss_final = history.history["loss"][-1]
        val_losses = history.history.get("val_loss", [])
        val_loss_final = val_losses[-1] if val_losses else None

        st.markdown("""
        - **Train loss** falling steadily = the model is learning.
        - **Val loss** rising while train loss falls = **overfitting** —
          the model is memorising training data rather than generalising.
        - Both converging at a low value = well-trained model.

        Try increasing **Dropout** or reducing **Units** if you see overfitting.
        """)

        if val_loss_final:
            divergence = (val_loss_final - train_loss_final) / (train_loss_final + 1e-9)
            if divergence > 0.5:
                st.warning(f"⚠️ **Val loss is significantly higher than train loss** ({divergence:.0%} gap). This suggests overfitting. Try increasing Dropout or reducing layer Units.")
            elif divergence < 0.1:
                st.success("✅ **Train and val loss are close.** The model generalises well.")
            else:
                st.info("ℹ️ **Slight gap between train and val loss** , a normal amount of overfitting. Watch this if you increase model complexity.")

    with st.expander("🔎 Raw Predictions vs Actual"):
        results_df = pd.DataFrame({
            "Date":          test_dates[:len(y_actual)],
            "Actual ($)":    np.round(y_actual, 2),
            "Predicted ($)": np.round(y_pred, 2),
            "Error ($)":     np.round(y_actual - y_pred, 2)
        })
        st.dataframe(results_df, width='stretch')

    # Suggestions
    st.markdown("---")
    st.markdown("### Suggestions Based on Your Results")

    suggestions = []
    if mape > 7:
        suggestions.append("🔧 **High MAPE:** Try increasing the lookback window (e.g. to 60–90 days) to give the model more historical context.")
    if val_loss_final and (val_loss_final - train_loss_final) / (train_loss_final + 1e-9) > 0.5:
        suggestions.append("🔧 **Overfitting detected:** Increase the dropout rate on each layer (try 0.3–0.4) or reduce the number of units per layer.")
    if epochs_used == epochs and not early_stopping:
        suggestions.append("🔧 **Training ran for the full epoch count:** Enable Early Stopping so training halts at the optimal point automatically.")
    if len(layer_configs) == 1 and mape > 5:
        suggestions.append("🔧 **Single-layer network:** Adding a second LSTM layer may help the model capture more complex patterns.")
    if learning_rate >= 0.005 and val_loss_final and val_loss_final > train_loss_final * 1.3:
        suggestions.append("🔧 **High learning rate:** If the loss curve is unstable, try reducing the learning rate to 0.001 or 0.0005.")

    if suggestions:
        for s in suggestions:
            st.markdown(s)
    else:
        st.success("✅ Your current configuration looks well-balanced. If you want to experiment further, try changing the lookback window or adding another LSTM layer to see how performance changes.")

else:
    st.info("👈 Configure your LSTM in the sidebar, then click **▶ Train Model** to begin.")
    st.markdown("""
    **Not sure where to start?** Try these defaults:
    - Lookback Window: **60 days**
    - 2 LSTM Layers with **64 units** each
    - Dropout: **0.2** on each layer
    - Learning Rate: **0.001**
    - Epochs: **50** with Early Stopping enabled

    Then read the explanations above to understand what each setting does before you experiment.
    """)
lab_specific_questions = [
    {
        "id": "q1",
        "type": "radio",
        "question": "What scaling technique is applied to the closing prices before training?",
        "options": ["Standard Scaling (Z-score)", "Min-Max Scaling", "Log Transformation", "No scaling is applied"],
        "answer": "Min-Max Scaling"
    },
    {
        "id": "q2",
        "type": "radio",
        "question": "What does the 'Forget Gate' in an LSTM unit do?",
        "options": [
            "Stores new information from the current time step",
            "Passes the cell state to the next layer",
            "Decides which information from the past should be erased",
            "Controls the learning rate during training"
        ],
        "answer": "Decides which information from the past should be erased"
    },
    {
        "id": "q3",
        "type": "radio",
        "question": "Why must time series data never be randomly shuffled before splitting into train and test sets?",
        "options": [
            "It increases training time unnecessarily",
            "Shuffling causes data leakage , the model could see future data during training",
            "The LSTM architecture requires sequential batch ordering",
            "MinMaxScaler cannot handle shuffled data"
        ],
        "answer": "Shuffling causes data leakage , the model could see future data during training"
    },
    {
        "id": "q4",
        "type": "radio",
        "question": "What does Early Stopping do when validation loss fails to improve for 10 consecutive epochs?",
        "options": [
            "Resets the model weights to random initialisation",
            "Doubles the learning rate to escape the plateau",
            "Halts training and restores weights from the best epoch",
            "Reduces the batch size and continues training"
        ],
        "answer": "Halts training and restores weights from the best epoch"
    },
    {
        "id": "q5",
        "type": "radio",
        "question": "If validation loss is rising while training loss continues to fall, what does this indicate?",
        "options": [
            "The learning rate is too low",
            "The model is underfitting the training data",
            "The lookback window is too short",
            "The model is overfitting , memorising rather than generalising"
        ],
        "answer": "The model is overfitting , memorising rather than generalising"
    },
     {
        "id": "sq1",
        "type": "text",
        "question": "What is MAPE and what does a value below 5% indicate about the model's performance?"
    },
    {
        "id": "sq2",
        "type": "text",
        "question": "Explain in your own words why dropout is disabled during prediction but enabled during training."
    },
    {
        "id": "sq3",
        "type": "text",
        "question": "Describe one key limitation of this LSTM model that could cause it to fail on real-world stock data."
    }
]
verify_create_quiz(
        "Neural Networks",
        lab_specific_questions
)