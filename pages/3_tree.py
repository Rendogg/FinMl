import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
from utilFunctions import sidebar, verify_create_quiz,apply_theme


# Page Config
st.set_page_config(page_title="Company Health Lab", page_icon="🏦", layout="wide")
sidebar("Tree")
apply_theme()

# Synthetic Dataset Generation
@st.cache_data
def generate_company_data(n=400, seed=42):
    """
    Generate realistic synthetic company fundamentals.
    Labels are deterministic based on rules inspired by the Altman Z-Score.
    """
    rng = np.random.default_rng(seed)

    pe_ratio        = rng.uniform(0, 60, n)        # Price-to-Earnings
    debt_to_equity  = rng.uniform(0, 5, n)         # D/E ratio
    current_ratio   = rng.uniform(0.3, 4.0, n)     # Current Assets / Current Liabilities
    roe             = rng.uniform(-0.3, 0.5, n)    # Return on Equity
    profit_margin   = rng.uniform(-0.2, 0.4, n)    # Net Profit Margin
    revenue_growth  = rng.uniform(-0.3, 0.6, n)    # YoY Revenue Growth
    interest_cover  = rng.uniform(-2, 20, n)       # EBIT / Interest Expense

    # Rule-based labelling (mimics financial distress models)
    score = (
        (pe_ratio < 30).astype(int) * 1 +
        (debt_to_equity < 1.5).astype(int) * 2 +
        (current_ratio > 1.5).astype(int) * 2 +
        (roe > 0.1).astype(int) * 2 +
        (profit_margin > 0.05).astype(int) * 2 +
        (revenue_growth > 0.05).astype(int) * 1 +
        (interest_cover > 3).astype(int) * 2
    )

    # Add noise
    score = score + rng.integers(-1, 2, n)
    score = np.clip(score, 0, 12)

    labels = np.where(score >= 9, "Healthy",
             np.where(score >= 5, "Caution", "Distressed"))

    build_df = pd.DataFrame({
        "P/E Ratio":        np.round(pe_ratio, 2),
        "Debt/Equity":      np.round(debt_to_equity, 2),
        "Current Ratio":    np.round(current_ratio, 2),
        "ROE":              np.round(roe, 3),
        "Profit Margin":    np.round(profit_margin, 3),
        "Revenue Growth":   np.round(revenue_growth, 3),
        "Interest Coverage":np.round(interest_cover, 2),
        "Health":           labels
    })
    return build_df


# UI
st.title("🏦 Company Health Scoring Lab")
st.markdown("""
This lab uses a **Decision Tree classifier** to score companies as **Healthy**, **Caution**, or **Distressed**
based on real financial ratios. Explore how tree structure, depth, and pruning affect the model
then compare it against a Random Forest and a naïve baseline.
""")
st.divider()

# Data
df = generate_company_data()

features = [
    "P/E Ratio", "Debt/Equity", "Current Ratio",
    "ROE", "Profit Margin", "Revenue Growth", "Interest Coverage"]
X = df[features]
y = df["Health"]

# Dataset Overview
with st.expander("Step 1 The Dataset & Financial Ratios", expanded=False):
    st.markdown("""
    ### What data are we using?

    We use **400 synthetic companies** with 7 fundamental financial ratios, labelled using
    a scoring system inspired by the **Altman Z-Score**. An academic model from 1968 that
    remains widely used in credit risk and distress prediction.

    | Feature | What it measures | Healthy signal |
    |---------|-----------------|----------------|
    | **P/E Ratio** | Price paid per £1 of earnings | < 30 (not wildly overvalued) |
    | **Debt/Equity** | Financial leverage | < 1.5 (not over-leveraged) |
    | **Current Ratio** | Short-term liquidity | > 1.5 (can cover near-term) |
    | **ROE** | Profitability relative to equity | > 10% |
    | **Profit Margin** | % of revenue kept as profit | > 5% |
    | **Revenue Growth** | YoY top-line expansion | > 5% |
    | **Interest Coverage** | Ability to service debt (EBIT/Interest) | > 3× |

    Labels: 🟢 **Healthy** 🟡 **Caution** 🔴 **Distressed**
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.sample(10, random_state=1).reset_index(drop=True), width='stretch')
    with col2:
        counts = df["Health"].value_counts()
        colors = {"Healthy": "#22c55e", "Caution": "#eab308", "Distressed": "#ef4444"}
        fig_pie = go.Figure(go.Pie(
            labels=counts.index,
            values=counts.values,
            marker_colors=[colors[l] for l in counts.index],
            hole=0.4
        ))
        fig_pie.update_layout(
            template="plotly_dark", height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True
        )
        st.plotly_chart(fig_pie, width='stretch')

# Sidebar Hyperparameters
st.sidebar.markdown("---")
st.sidebar.header("🌳 Decision Tree Parameters")

criterion = st.sidebar.selectbox(
    "Splitting criterion",
    ["gini", "entropy", "log_loss"],
    help="How the tree decides where to split. Gini is fast and works well; entropy/log_loss are similar but a bit more precise."
)
max_depth = st.sidebar.slider(
    "max_depth Maximum tree depth",
    min_value=1, max_value=15, value=4,
    help="How deep the tree can grow. Small = simple model (may miss patterns). Large = more complex (may overfit)."
)
min_samples_split = st.sidebar.slider(
    "min_samples_split",
    min_value=2, max_value=50, value=10,
    help="Minimum number of samples required to split an internal node. Higher values make the tree more cautious."
)
min_samples_leaf = st.sidebar.slider(
    "min_samples_leaf",
    min_value=1, max_value=30, value=5,
    help="Minimum samples required at a leaf node. Prevents tiny, noise-fitting leaves."
)
train_pct = st.sidebar.slider("Train set size (%)", 50, 90, 75)

st.sidebar.markdown("---")
st.sidebar.header("🌲 Random Forest Parameters")
rf_n_estimators = st.sidebar.slider("n_estimators (trees)", 10, 300, 100, step=10)
rf_max_depth    = st.sidebar.slider("RF max_depth", 1, 15, 6)


# Train / Test Split

split_idx = int(len(df) * (train_pct / 100))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# How Decision Trees Work

with st.expander("🌳 Step 2 How Decision Trees Work", expanded=False):
    st.markdown(f"""
    ### Recursive binary splitting

    A Decision Tree works by repeatedly asking **yes/no questions** about features(our financial ratios),
    splitting the data into purer and purer subsets until it reaches a stopping condition.

    Each **internal node** poses a question: *"Is Debt/Equity > 1.5?"*  
    Each **branch** follows the True or False path.  
    Each **leaf node** gives a final class label.

    #### Splitting criteria (you selected: `{criterion}`)

    | Criterion | Formula | Intuition |
    |-----------|---------|-----------|
    | **Gini** | 1 − Σpᵢ² | Probability of misclassifying a random sample |
    | **Entropy** | −Σpᵢ log₂(pᵢ) | Information content / uncertainty |
    | **Log Loss** | Same as entropy | Focuses on how well predicted probabilities match reality|

    The algorithm chooses the split that **reduces impurity the most** at each step.
    In essence for each node the criterion  determines the best feature and threshold the feature
    is compared against. I.E  *"Is Debt/Equity > 1.5?"*
      
    #### The Bias–Variance tradeoff

    | `max_depth` | Bias | Variance | Risk |
    |-------------|------|----------|------|
    | Very shallow (1–2) | High | Low | Underfitting can miss real patterns |
    | Moderate (3–6) | Balanced | Balanced | Usually optimal zone |
    | Very deep (10+) | Low | High | Overfitting memorises training noise |

    > **Pruning parameters** (`min_samples_split`, `min_samples_leaf`) stop the tree from
    > growing branches that only serve a handful of samples, a major source of overfitting.
    """)


# Train models
with st.spinner("Training models…"):
    dt = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred_bl = baseline.predict(X_test)


# Tree Visualisation

st.subheader("🌳 Decision Tree Structure")

with st.expander("How to read this tree", expanded=False):
    st.markdown("""
    Each box shows:
    - **The split condition** tested at that node (e.g. `Debt/Equity <= 1.5`)
    - **Gini / entropy** impurity at that node (0 = perfectly pure, 1 = maximally mixed)
    - **Samples** how many training examples reached this node
    - **Value** class counts [Caution, Distressed, Healthy]
    - **Class** the majority class label

    Follow the **left branch** when the condition is **True**, right when **False**.
    A pure leaf (gini = 0) means all samples there belong to one class.
    """)

fig_tree, ax_tree = plt.subplots(figsize=(max(14, max_depth * 3), max(6, max_depth * 2)))
fig_tree.patch.set_facecolor("#0f172a")
ax_tree.set_facecolor("#0f172a")

class_names = sorted(y.unique().tolist())
plot_tree(
    dt,
    feature_names=features,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax_tree,
    impurity=True,
    proportion=False
)
plt.tight_layout()
st.pyplot(fig_tree, width='stretch')
plt.close()

# Text rules
with st.expander("View tree as text rules", expanded=False):
    rules = export_text(dt, feature_names=features)
    st.code(rules, language="text")


# Metrics

st.subheader("Model Performance")

with st.expander("Understanding the metrics", expanded=False):
    st.markdown("""
    | Metric | Definition | Why it matters here |
    |--------|-----------|---------------------|
    | **Accuracy** | % of correct predictions overall | Baseline measure, but misleading if classes are imbalanced |
    | **Precision** | Of all predicted Distressed, how many truly were? | False positives = unnecessary alarm |
    | **Recall** | Of all truly Distressed, how many did we catch? | False negatives = missed risk **most costly in finance** |
    | **F1 Score** | Harmonic mean of precision & recall | Balances both concerns |

    > In a credit risk context, **recall for "Distressed"** is typically the most important metric
    > missing a troubled company is more dangerous than over-flagging a healthy one.
    """)

acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_bl = accuracy_score(y_test, y_pred_bl)

col1, col2, col3 = st.columns(3)
col1.metric("🌳 Decision Tree Accuracy",  f"{acc_dt:.1%}",
            delta=f"{acc_dt - acc_bl:+.1%} vs baseline")
col2.metric("🌲 Random Forest Accuracy",  f"{acc_rf:.1%}",
            delta=f"{acc_rf - acc_bl:+.1%} vs baseline")
col3.metric("📊 Baseline (Majority Class)", f"{acc_bl:.1%}")

# Per-class report
st.markdown("#### Per-class Classification Report")
report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

report_rows = []
for cls in ["Caution", "Distressed", "Healthy"]:
    if cls in report_dt:
        report_rows.append({
            "Class": cls,
            "DT Precision": f"{report_dt[cls]['precision']:.2f}",
            "DT Recall":    f"{report_dt[cls]['recall']:.2f}",
            "DT F1":        f"{report_dt[cls]['f1-score']:.2f}",
            "RF Precision": f"{report_rf[cls]['precision']:.2f}",
            "RF Recall":    f"{report_rf[cls]['recall']:.2f}",
            "RF F1":        f"{report_rf[cls]['f1-score']:.2f}",
        })
st.dataframe(pd.DataFrame(report_rows), width='stretch', hide_index=True)


# Confusion Matrix

st.subheader("🔲 Confusion Matrices")

with st.expander("ℹ️ How to read a confusion matrix", expanded=False):
    st.markdown("""
    Rows = **actual** class. Columns = **predicted** class.
    - **Diagonal** cells (top-left to bottom-right) = correct predictions ✅
    - **Off-diagonal** cells = errors ❌
    - A large off-diagonal value in the Distressed row means the model missed many troubled companies.
    """)

col_cm1, col_cm2 = st.columns(2)

def plot_cm(y_true, y_pred, title, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale=[[0, "#0f172a"], [0.5, "#1e3a5f"], [1, "#00d4ff"]],
        text=cm.astype(str),
        texttemplate="%{text}",
        showscale=False
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=320,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig

classes = ["Caution", "Distressed", "Healthy"]
with col_cm1:
    st.plotly_chart(plot_cm(y_test, y_pred_dt, "Decision Tree", classes), width='stretch')
with col_cm2:
    st.plotly_chart(plot_cm(y_test, y_pred_rf, "Random Forest", classes), width='stretch')


# Feature Importance
st.subheader("📊 Feature Importance")

with st.expander("ℹ️ Decision Tree vs Random Forest importance", expanded=False):
    st.markdown("""
    #### Decision Tree
    Each feature's importance is the **total reduction in impurity** it contributes across all
    splits in the single tree, weighted by the proportion of samples reaching each node.

    #### Random Forest Mean decrease in impurity
    Averages importance **across all trees** in the forest. This is more stable than a
    single tree's importance because individual trees overfit to different subsets of data.

    > **Key insight**: If a feature is important to the Decision Tree but not to the Random Forest
    > (or vice versa), that feature may be overfitted to specific training samples.
    """)

imp_dt = pd.DataFrame({"Feature": features, "Importance": dt.feature_importances_,
                        "Model": "Decision Tree"})
imp_rf = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_,
                        "Model": "Random Forest"})
imp_all = pd.concat([imp_dt, imp_rf]).sort_values("Importance", ascending=True)

fig_imp = px.bar(
    imp_all, x="Importance", y="Feature", color="Model",
    barmode="group", orientation="h",
    color_discrete_map={"Decision Tree": "#00d4ff", "Random Forest": "#f59e0b"},
    template="plotly_dark", height=360
)
fig_imp.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title="")
st.plotly_chart(fig_imp, width='stretch')


# Depth vs Accuracy

st.subheader("🔍 Depth vs Accuracy (Bias–Variance Tradeoff)")

with st.expander("ℹ️ What does this chart show?", expanded=False):
    st.markdown("""
    This chart trains the Decision Tree at **every depth from 1 to 15** and plots both
    training and test accuracy.

    - **Training accuracy** always increases with depth , the tree can memorise more.
    - **Test accuracy** peaks at an optimal depth, then falls as the model overfits.
    - The **gap** between training and test accuracy is a direct measure of overfitting.

    The optimal depth is usually where test accuracy peaks before the gap widens significantly.
    Your current `max_depth` slider is marked with a vertical line.
    """)

depths       = list(range(1, 16))
train_accs   = []
test_accs    = []

for d in depths:
    m = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=d,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    m.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, m.predict(X_train)))
    test_accs.append(accuracy_score(y_test,  m.predict(X_test)))

fig_depth = go.Figure()
fig_depth.add_trace(go.Scatter(
    x=depths, y=train_accs, name="Train Accuracy",
    mode="lines+markers", line=dict(color="#00d4ff", width=2)
))
fig_depth.add_trace(go.Scatter(
    x=depths, y=test_accs, name="Test Accuracy",
    mode="lines+markers", line=dict(color="#f59e0b", width=2)
))
fig_depth.add_vline(
    x=max_depth, line_dash="dash", line_color="#ef4444"
)
fig_depth.add_annotation(
    x=max_depth, y=max(test_accs) * 0.97,
    text=f"  Current depth={max_depth}",
    showarrow=False, font=dict(color="#ef4444", size=11)
)
fig_depth.update_layout(
    template="plotly_dark",
    xaxis_title="max_depth",
    yaxis_title="Accuracy",
    yaxis_tickformat=".0%",
    hovermode="x unified",
    height=380,
    margin=dict(t=10, b=20)
)
st.plotly_chart(fig_depth, width='stretch')

# Model Comparison Summary
st.subheader("⚖️ Model Comparison")

with st.expander("Decision Tree vs Random Forest vs Baseline", expanded=False):
    st.markdown("""
    | Model | Strength | Weakness |
    |-------|----------|----------|
    | **Majority Baseline** | Zero effort | Always predicts the most common class useless for minorities |
    | **Decision Tree** | Fully interpretable, shows exact rules | Unstable small data changes can flip the whole structure |
    | **Random Forest** | More accurate, stable, robust to outliers | Black box time consuming to inspect individual trees |

    #### Why does Random Forest usually win?

    It trains **many trees on random subsets** (bagging) and averages their votes.
    The variance of individual trees cancels out, leaving a lower-variance ensemble.
    The cost is interpretability you can examine feature importances but not a single decision path.

    
    """)

comparison = pd.DataFrame({
    "Model":    ["Majority Baseline", "Decision Tree", "Random Forest"],
    "Accuracy": [f"{acc_bl:.1%}", f"{acc_dt:.1%}", f"{acc_rf:.1%}"],
    "Interpretable": ["✅", "✅", "❌"],
    "Stable":        ["✅", "❌", "✅"],
    "Notes": [
        "Always predicts most frequent class",
        f"depth={max_depth}, criterion={criterion}",
        f"{rf_n_estimators} trees, depth={rf_max_depth}"
    ]
})
st.dataframe(comparison, width='stretch', hide_index=True)
# Score a Custom Company
st.divider()
st.subheader("🔬 Score a Custom Company")
st.markdown("Enter your own financial ratios below and see what the models predict.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    inp_pe = st.number_input("P/E Ratio", 0.0, 100.0, 18.0, step=0.5)
    inp_de = st.number_input("Debt/Equity", 0.0, 10.0,  1.2, step=0.1)
with c2:
    inp_cr = st.number_input("Current Ratio", 0.1, 8.0,   1.8, step=0.1)
    inp_roe = st.number_input("ROE", -0.5, 1.0,   0.12, step=0.01, format="%.2f")
with c3:
    inp_pm = st.number_input("Profit Margin", -0.5, 1.0,   0.08, step=0.01, format="%.2f")
    inp_rg = st.number_input("Revenue Growth", -0.5, 2.0,   0.10, step=0.01, format="%.2f")
with c4:
    inp_ic = st.number_input("Interest Coverage", -5.0, 30.0,  5.0, step=0.5)

custom = pd.DataFrame([[inp_pe, inp_de, inp_cr, inp_roe, inp_pm, inp_rg, inp_ic]],
                      columns=features)

pred_dt_custom = dt.predict(custom)[0]
pred_rf_custom = rf.predict(custom)[0]
prob_dt = dt.predict_proba(custom)[0]
prob_rf = rf.predict_proba(custom)[0]
class_order = dt.classes_

label_colors = {"Healthy": "🟢", "Caution": "🟡", "Distressed": "🔴"}

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown(f"### 🌳 Decision Tree says: {label_colors.get(pred_dt_custom,'')} **{pred_dt_custom}**")
    prob_df_dt = pd.DataFrame({"Class": class_order, "Probability": prob_dt})
    fig_prob_dt = go.Figure(go.Bar(
        x=prob_df_dt["Class"], y=prob_df_dt["Probability"],
        marker_color=["#eab308" if c=="Caution" else "#ef4444" if c=="Distressed" else "#22c55e"
                      for c in prob_df_dt["Class"]]
    ))
    fig_prob_dt.update_layout(
        template="plotly_dark", yaxis_tickformat=".0%",
        height=240, margin=dict(t=10, b=10), yaxis_title="Probability"
    )
    st.plotly_chart(fig_prob_dt, width='stretch')

with col_r2:
    st.markdown(f"### 🌲 Random Forest says: {label_colors.get(pred_rf_custom,'')} **{pred_rf_custom}**")
    prob_df_rf = pd.DataFrame({"Class": class_order, "Probability": prob_rf})
    fig_prob_rf = go.Figure(go.Bar(
        x=prob_df_rf["Class"], y=prob_df_rf["Probability"],
        marker_color=["#eab308" if c=="Caution" else "#ef4444" if c=="Distressed" else "#22c55e"
                      for c in prob_df_rf["Class"]]
    ))
    fig_prob_rf.update_layout(
        template="plotly_dark", yaxis_tickformat=".0%",
        height=240, margin=dict(t=10, b=10), yaxis_title="Probability"
    )
    st.plotly_chart(fig_prob_rf, width='stretch')
    
# Footer
st.divider()
st.info("""
**Lab Note Why not use real data?**  
Real company fundamentals require either a paid data provider (Bloomberg, Refinitiv) or web scraping
with significant cleaning overhead. This lab uses **synthetically generated data** built to match
realistic ratio distributions and labelled using rules derived from the Altman Z-Score framework,
so all algorithmic insights remain valid only the company names are missing.
""")

# Quiz Section
lab_specific_questions = [
{
    "id": "q1",
    "type": "radio",
    "question": "If a Decision Tree uses the Altman Z-Score as a parameter, what is it most likely trying to predict?",
    "options": ["Next day's closing price", "The probability of a company going bankrupt", "The total dividend payout",
                "The volatility of the sector"],
    "answer": "The probability of a company going bankrupt"
},
{
    "id": "q2",
    "type": "radio",
    "question": "What does a Leaf Node(tail end of the tree) in a Decision Tree represent?",
    "options": ["A question about a attribute", "A branch that was pruned", "The final classification or predicted outcome",
                "The volatility of the sector"],
    "answer": "The final classification or predicted outcome"
},
{
    "id": "q3",
    "type": "radio",
    "question": "What is Pruning in a Decision Tree model?",
    "options": ["Adding more features to the data", "Cutting back branches to prevent the tree from becoming too complex (overfitting)", 
                "Increasing the depth of the tree to capture every detail",
                "Changing the Z-Score formula"],
    "answer": "Cutting back branches to prevent the tree from becoming too complex (overfitting)"
},
{
    "id": "q4",
    "type": "radio",
    "question": "Decision Trees are often called White Box models. What does this mean?",
    "options": ["They are invisible to the user", 
                "The logic (if-then rules) is easy for humans to see and understand", 
                "They only work with positive numbers",
                "They are highly complex and impossible to explain"],
    "answer": "Cutting back branches to prevent the tree from becoming too complex (overfitting)"
},
]
verify_create_quiz(
    "Decision Trees and Random Forests",
    lab_specific_questions
)