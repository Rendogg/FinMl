import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import yfinance as yf
import streamlit as st
from utilFunctions import sidebar
from utilFunctions import get_data, verify_create_quiz, apply_theme
sidebar("Clustering")
data = get_data() if st.session_state.get("current_mode") != "Stock Comparison" else None

# PAGE CONFIG
st.set_page_config(page_title="Clustering Lab | FinML Lab", layout="wide")
apply_theme()


# BACKEND HELPERS
@st.cache_data(show_spinner=False)
def fetch_multi_stock_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for multiple tickers.
    Used for the cross-stock clustering mode.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw.dropna(axis=1, how="all", inplace=True)
    raw.dropna(inplace=True)
    return raw


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-day features from OHLCV data for Market Day clustering.

    Features:
        daily_return   percentage change in closing price
        intraday_range (High - Low) / Open  → good measure for volatility
        volume_change  percentage change in volume
    """
    feat = pd.DataFrame(index=df.index)
    feat["daily_return"]   = df["Close"].pct_change() * 100
    feat["intraday_range"] = (df["High"] - df["Low"]) / df["Open"] * 100
    feat["volume_change"]  = df["Volume"].pct_change() * 100
    feat.dropna(inplace=True)
    return feat


def build_stock_features(close_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-stock features from a matrix of close prices.

    Features:
        mean_return     average daily return over the period
        volatility      standard deviation of daily returns
        max_drawdown    largest peak-to-trough decline
    """
    returns = close_prices.pct_change().dropna()
    feat = pd.DataFrame(index=close_prices.columns)
    feat["mean_return"]  = returns.mean() * 100
    feat["volatility"]   = returns.std() * 100
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    feat["max_drawdown"] = drawdown.min() * 100
    return feat


def run_kmeans(features: pd.DataFrame, k: int, selected_features: list[str]):
    """
    Scale selected features, run K-Means, and return labels + scaled array.
    """
    X = features[selected_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return labels, X_scaled, X.index, model


def compute_elbow(X_scaled: np.ndarray, max_k: int = 10) -> dict:
    """Return inertia and silhouette scores for k = 2 … max_k."""
    ks, inertias, silhouettes = [], [], []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        ks.append(k)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return {"k": ks, "inertia": inertias, "silhouette": silhouettes}

# PLOTTING HELPERS
CLUSTER_PALETTE = px.colors.qualitative.Bold

def plot_elbow(elbow_data: dict, chosen_k: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=elbow_data["k"], y=elbow_data["inertia"],
        mode="lines+markers", name="Inertia",
        line=dict(color="#4C9BE8", width=2),
        marker=dict(size=7)
    ))
    fig.add_vline(
        x=chosen_k, line_dash="dash", line_color="#F4845F",
        annotation_text=f"Your K = {chosen_k}",
        annotation_position="top right"
    )
    fig.update_layout(
        title="Elbow Plot Inertia vs Number of Clusters",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (Within-cluster sum of squares)",
        template="plotly_dark",
        height=350
    )
    return fig


def plot_silhouette(elbow_data: dict, chosen_k: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=elbow_data["k"], y=elbow_data["silhouette"],
        marker_color=[
            "#F4845F" if k == chosen_k else "#4C9BE8"
            for k in elbow_data["k"]
        ],
        name="Silhouette Score"
    ))
    fig.update_layout(
        title="Silhouette Score vs Number of Clusters",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score (higher = better defined)",
        template="plotly_dark",
        height=350
    )
    return fig


def plot_clusters_2d(X_scaled: np.ndarray, labels: np.ndarray,
                     point_names, feat_names: list[str], mode: str) -> go.Figure:
    """
    Reduce to 2D via PCA when more than 2 features, else plot directly.
    """
    if X_scaled.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        x_label = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"
    else:
        coords = X_scaled
        x_label, y_label = feat_names[0], feat_names[1]

    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Cluster": [f"Cluster {l+1}" for l in labels],
        "Label": list(point_names)
    })

    hover = "Label" if mode == "stocks" else "Label"
    fig = px.scatter(
        df_plot, x="x", y="y", color="Cluster",
        hover_name=hover,
        color_discrete_sequence=CLUSTER_PALETTE,
        title="Cluster Visualisation (PCA-reduced to 2D)" if X_scaled.shape[1] > 2
              else "Cluster Visualisation",
        template="plotly_dark",
        height=450
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    return fig


def plot_cluster_profiles(features: pd.DataFrame, labels: np.ndarray,
                          selected_features: list[str]) -> go.Figure:
    """
    Radar / bar chart showing mean feature values per cluster.
    Helps the user interpret what each cluster 'means'.
    """
    df = features[selected_features].copy()
    df["Cluster"] = [f"Cluster {l+1}" for l in labels]
    profile = df.groupby("Cluster")[selected_features].mean().reset_index()

    fig = go.Figure()
    for _, row in profile.iterrows():
        fig.add_trace(go.Bar(
            name=row["Cluster"],
            x=selected_features,
            y=row[selected_features].values,
        ))
    fig.update_layout(
        barmode="group",
        title="Average Feature Values per Cluster",
        xaxis_title="Feature",
        yaxis_title="Mean Value",
        template="plotly_dark",
        height=380
    )
    return fig

# SIDEBAR Configuration

with st.sidebar:
    st.markdown("## ⚙️ Clustering Configuration")
    st.markdown("---")

    # Clustering Mode 
    mode = st.radio(
        "Clustering Mode",
        options=["Market Days", "Stock Comparison"],
        help=(
            "**Market Days**: Cluster individual trading days for one stock by "
            "how they behaved (return, volatility, volume change).\n\n"
            "**Stock Comparison**: Cluster multiple stocks by their overall "
            "behaviour across the selected period."
        )
    )

    st.markdown("---")
    if mode == "Market Days":
        ticker = st.session_state.get("ticker", "AAPL")
        tickers = [ticker]

    else:
        raw_tickers = st.text_input(
        "Ticker Symbols (comma-separated)",
        value="AAPL, MSFT, GOOG, AMZN, TSLA, JPM, GS, XOM, JNJ, PFE, WMT, NFLX, META, NVDA, BA"
    )
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

    # Feature Selection
    st.markdown("### Feature Selection")
    st.caption("Choose which aspects of the data the model should group by.")

    if mode == "Market Days":
        all_features = ["daily_return", "intraday_range", "volume_change"]
        feature_labels = {
            "daily_return":   "Daily Return (%)",
            "intraday_range": "Intraday Range (% volatility)",
            "volume_change":  "Volume Change (%)"
        }
    else:
        all_features = ["mean_return", "volatility", "max_drawdown"]
        feature_labels = {
            "mean_return":  "Mean Daily Return (%)",
            "volatility":   "Volatility (std dev of returns)",
            "max_drawdown": "Maximum Drawdown (%)"
        }

    selected_features = [
        f for f in all_features
        if st.checkbox(feature_labels[f], value=True)
    ]

    st.markdown("---")

    # K-Means Hyperparameters
    st.markdown("### Model Hyperparameters")

    k = st.slider(
        "Number of Clusters (K)",
        min_value=2, max_value=10, value=3,
        help=(
            "K tells the algorithm how many groups to find. "
            "There is no single correct answer use the Elbow Plot below "
            "to help you decide."
        )
    )

    show_elbow = st.checkbox("Show Elbow & Silhouette Plots", value=True)

# UI

st.title("🔵 K-Means Clustering Lab")
st.caption("Discover hidden groupings in stock market data no labels required.")

# What is Clustering?
with st.expander("📖 What is Clustering?", expanded=True):
    st.markdown("""
    **Clustering** is a type of *unsupervised* machine learning, meaning the 
    algorithm finds patterns in data without being told what to look for.
    There are no predefined labels like "good" or "bad". Instead, the algorithm 
    groups data points that are *similar to each other* and *different from 
    other groups*.

    > Imagine sorting a deck of cards without being told 
    > the suits. You'd notice some cards share similar symbols and group them 
    > naturally.

    **Why is this useful in finance?**
    - Identifying stocks that *move together* helps with **portfolio diversification** 
      owning stocks from different clusters reduces risk.
    - Grouping market days reveals **market regimes**: calm trending days vs 
      high-volatility crash days vs low-volume drifting days.
    - Professional traders use clustering to inform **sector rotation** strategies(Moving between industries).
    """)

# How K-Means Works 
with st.expander("⚙️ How Does K-Means Work?"):
    st.markdown("""
    K-Means is the most widely used clustering algorithm. Here's how it works 
    step by step:

    1. **Choose K** : You decide how many clusters you want (this is the K slider 
       in the sidebar).
    2. **Place centroids** : K random points are placed in the data as initial 
       cluster centres.
    3. **Assign points** : Every data point is assigned to its nearest centroid.
    4. **Move centroids** : Each centroid moves to the average position of all 
       points assigned to it.
    5. **Repeat** : Steps 3 and 4 repeat until the assignments stop changing.

    The algorithm minimises **inertia** : the total distance between each point 
    and its cluster's centre. The lower the inertia, the tighter the clusters.

    > **Important:** K-Means requires you to specify K in advance. The 
    > **Elbow Plot** (below) will help you choose a sensible value.
    """)

# About the Features 
with st.expander("📊 About the Features"):
    if mode == "Market Days":
        st.markdown(f"""
        In **Market Days** mode, each data point is a single trading day for 
        **{tickers[0]}**. The algorithm groups days that *behaved similarly*.

        | Feature | What it measures |
        |---|---|
        | **Daily Return (%)** | How much the closing price changed vs. the previous day |
        | **Intraday Range (%)** | How wide the price swung from low to high within the day a proxy for volatility |
        | **Volume Change (%)** | Whether unusually high or low trading volume occurred |

        Days with large negative returns *and* high intraday range *and* high 
        volume might cluster together as **panic selling days**. 
        Calm, low-volume days with small price changes form a very different cluster.
        """)
    else:
        st.markdown("""
        In **Stock Comparison** mode, each data point is a stock. 
        The algorithm groups stocks that *behaved similarly over the period*.

        | Feature | What it measures |
        |---|---|
        | **Mean Daily Return (%)** | Average day-to-day price growth |
        | **Volatility** | Standard deviation of daily returns how erratic the stock was |
        | **Maximum Drawdown (%)** | The worst peak-to-trough decline a key risk metric |

        High-volatility, high-return stocks tend to cluster separately from 
        stable, dividend-like stocks even if you don't label them that way in advance.
        """)

st.markdown("---")

# Load Data & Run Model 
if not selected_features:
    st.warning("⚠️ Please select at least one feature in the sidebar to continue.")
    st.stop()

with st.spinner("Fetching data from Yahoo Finance…"):
    try:
        if mode == "Market Days":
            raw_df   = data.copy()
            raw_df.set_index("Date", inplace=True)
            features = build_daily_features(raw_df)
        else:
            if len(tickers) < k:
                st.error(f"You need at least **{k} stocks** to form **{k} clusters**. "
                         f"Add more tickers or reduce K.")
                st.stop()
            close_df = fetch_multi_stock_data(tickers,str(st.session_state.start_date),str(st.session_state.end_date))
            features = build_stock_features(close_df)
    except Exception as e:
        st.error(f"Could not retrieve data: {e}")
        st.stop()

# Validate enough data points. Likely not needed but good to have it handled
if len(features) < k:
    st.error(f"Not enough data points ({len(features)}) to form {k} clusters. "
             "Try extending your date range or reducing K.")
    st.stop()

# Run K-Means
labels, X_scaled, valid_index, km_model = run_kmeans(features, k, selected_features)
features_valid = features.loc[valid_index]

# Elbow & Silhouette
if show_elbow:
    st.subheader("🔍 Choosing the Right K")
    st.markdown("""
    Before interpreting the clusters, it's worth understanding how to choose K.
    The two charts below offer different perspectives on cluster quality.
    """)

    with st.spinner("Computing elbow data…"):
        max_k_elbow = min(10, len(features) - 1)
        elbow_data  = compute_elbow(X_scaled, max_k=max_k_elbow)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_elbow(elbow_data, k), width='stretch')
        st.caption(
            "**Elbow Plot:** As K increases, inertia always falls. "
            "Look for the 'elbow' the point where adding more clusters "
            "gives diminishing returns. This is often a sensible K."
        )
    with col2:
        st.plotly_chart(plot_silhouette(elbow_data, k), width='stretch')
        st.caption(
            "**Silhouette Score:** Measures how well-separated the clusters are "
            "(range: -1 to 1). Higher is better. The highlighted bar is your "
            "currently selected K."
        )

    st.markdown("---")

# Cluster Visualisation
st.subheader("🗺️ Cluster Visualisation")
st.markdown(
    "Each point below represents a "
    + ("**trading day**" if mode == "Market Days" else "**stock**")
    + ". Points of the same colour belong to the same cluster. "
    "Hover over a point to see its label."
)

point_names = (
    features_valid.index.strftime("%Y-%m-%d") if mode == "Market Days"
    else valid_index
)
st.plotly_chart(
    plot_clusters_2d(X_scaled, labels, point_names, selected_features, mode),
    width='stretch'
)

# Cluster Profiles 
st.subheader("📋 What Does Each Cluster Mean?")
st.markdown(
    "This is the most important step translating the algorithm's output "
    "into financial understanding. The chart below shows the average feature "
    "values for each cluster."
)

features_valid = features_valid.copy()
features_valid["Cluster"] = [f"Cluster {l+1}" for l in labels]

st.plotly_chart(
    plot_cluster_profiles(features_valid, labels, selected_features),
    width='stretch'
)

# Cluster Summary Table
st.subheader("📊 Cluster Summary")
profile_table = (
    features_valid
    .groupby("Cluster")[selected_features]
    .agg(["mean", "std", "count"])
)
profile_table.columns = [" ".join(c) for c in profile_table.columns]
st.dataframe(profile_table.style.format("{:.2f}"), width='stretch')

with st.expander(
    "🔎 See Full Cluster Assignments"
    + (" (all trading days)" if mode == "Market Days" else " (all stocks)")
):
    display_df = features_valid.reset_index()
    display_df.columns = (
        ["Date"] + selected_features + ["Cluster"]
        if mode == "Market Days"
        else ["Ticker"] + selected_features + ["Cluster"]
    )
    st.dataframe(display_df, width='stretch')

st.markdown("---")

# Key Takeaways
with st.expander("🎓 Key Takeaways from This Lab", expanded=True):
    if mode == "Market Days":
        st.markdown(f"""
        You just used K-Means to cluster every trading day for **{tickers[0]}**
        into **{k} groups** based on how the market behaved that day.

        **What to take away:**
        - Clusters reveal **market regimes** recurring types of trading day 
          that the market cycles through.
        - By knowing which regime today resembles, a trader can adjust their 
          strategy accordingly.
        - Notice that you didn't tell the algorithm *what* the clusters should 
          be it found the structure itself. That's the power of unsupervised learning.

        **Try this:** Change K from 3 to 5 and see whether the clusters become 
        more granular. Does the Elbow Plot suggest 3 or 5 is a better fit?
        """)
    else:
        st.markdown(f"""
        You just grouped **{len(tickers)} stocks** into **{k} clusters** based
        on their behaviour over the selected period.

        **What to take away:**
        - Stocks in the same cluster tend to move together and carry similar risk 
          profiles. Holding only stocks from one cluster is **poorly diversified**.
        - Stocks in *different* clusters provide genuine diversification their 
          different behaviours mean they don't all fall at the same time.
        - This is the foundation of **factor investing** and **sector rotation** 
          strategies used by professional fund managers.

        **Try this:** Add or remove tickers and observe how the cluster 
        memberships change. Does TSLA always end up in a different cluster from JNJ?
        """)
# Lab Quiz  
lab_specific_questions = [
    {
        "id": "q1",
        "type": "radio",
        "question": "What type of machine learning is K-Means clustering?",
        "options": [
            "Supervised learning it learns from labelled examples",
            "Unsupervised learning it finds patterns without predefined labels",
            "Reinforcement learning it learns through rewards and penalties",
            "Semi-supervised learning it uses a mix of labelled and unlabelled data"
        ],
        "answer": "Unsupervised learning it finds patterns without predefined labels"
    },
    {
        "id": "q2",
        "type": "radio",
        "question": "What does 'inertia' measure in K-Means clustering?",
        "options": [
            "The number of iterations before the algorithm converges",
            "The silhouette score averaged across all clusters",
            "The total distance between each point and its cluster's centre",
            "The percentage of variance explained by the chosen K"
        ],
        "answer": "The total distance between each point and its cluster's centre"
    },
    {
        "id": "q3",
        "type": "radio",
        "question": "In Stock Comparison mode, what does 'Maximum Drawdown' measure?",
        "options": [
            "The average daily trading volume over the period",
            "The standard deviation of daily returns",
            "The difference between the highest and lowest closing price",
            "The worst peak-to-trough decline over the period"
        ],
        "answer": "The worst peak-to-trough decline over the period"
    },
    {
        "id": "q4",
        "type": "radio",
        "question": "Why is StandardScaler applied to features before running K-Means?",
        "options": [
            "To reduce the number of features passed to the algorithm",
            "To prevent features with larger numeric ranges from dominating distance calculations",
            "To convert categorical features into numerical values",
            "To remove outliers from the dataset before clustering"
        ],
        "answer": "To prevent features with larger numeric ranges from dominating distance calculations"
    },
    {
        "id": "q5",
        "type": "radio",
        "question": "What does a higher Silhouette Score indicate about the clusters?",
        "options": [
            "More clusters were used, increasing model complexity",
            "Inertia has been reduced to its minimum possible value",
            "The clusters are better separated and more internally cohesive",
            "The algorithm ran for more iterations before converging"
        ],
        "answer": "The clusters are better separated and more internally cohesive"
    },
    {
        "id": "sq1",
        "type": "text",
        "question": "Explain what the 'elbow' in an Elbow Plot represents and how you would use it to choose K."
    },
    {
        "id": "sq2",
        "type": "text",
        "question": "In Market Days mode, describe what a cluster of days with large negative returns, high intraday range, and high volume change might represent financially."
    },
    {
        "id": "sq3",
        "type": "text",
        "question": "Why does holding stocks from different clusters provide better portfolio diversification than holding stocks from the same cluster?"
    }
]
verify_create_quiz(
        "K-Means Clustering",
        lab_specific_questions)