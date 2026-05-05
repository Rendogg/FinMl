import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, datetime
from firebase_admin import db
from streamlit_option_menu import option_menu

# PAGE REGISTRY — add new pages here only
PAGES = [
    {"label": "Home",                  "icon": "house",          "path": "main.py",                          "show_stock_settings": False},
    {"label": "Introduction To Data",  "icon": "database",       "path": "pages/1_Introduction_To_Data.py",  "show_stock_settings": True},
    {"label": "Linear Regression Lab", "icon": "graph-up-arrow", "path": "pages/2_Linear_Regression.py",    "show_stock_settings": True},
    {"label": "Tree",                  "icon": "diagram-3",           "path": "pages/3_tree.py",                  "show_stock_settings": True},
    {"label": "Clustering",            "icon": "bounding-box-circles",       "path": "pages/4_Clustering.py",            "show_stock_settings": True},
    {"label": "Neural Networks",       "icon": "cpu",            "path": "pages/5_Neural Networks.py",       "show_stock_settings": True}
]


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_data(symbol, start, end, force_reload=False):
    return yf.download(symbol, start=start, end=end)


def get_data():
    if "data" not in st.session_state or st.session_state.data is None:
        with st.spinner("Fetching stock data..."):
            data = load_data(
                st.session_state.ticker,
                st.session_state.start_date,
                st.session_state.end_date,
                force_reload=st.session_state.get("force_reload", False),
            )

            if data is None or data.empty:
                st.error("Invalid ticker or no data available.")
                return None

            if isinstance(data.index, pd.DatetimeIndex):
                data.reset_index(inplace=True)
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

            st.session_state.data = data
            st.session_state.force_reload = False

    return st.session_state.data


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def _hide_default_nav():
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] { display: none; }
            div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) { padding-top: 0px; }
        </style>
    """, unsafe_allow_html=True)

def _nav_menu(current_index: int) -> str:
    return option_menu(
        menu_title=None,
        options=[p["label"] for p in PAGES],
        icons=[p["icon"] for p in PAGES],
        menu_icon="list",
        default_index=current_index,
        styles={
            "container": {
                "padding": "8px",
                "background-color": "#070B10",
                "border": "none",
            },

            "icon": {
                "color": "#00C9A7",
                "font-size": "18px",
            },

            "nav-link": {
                "font-size": "15px",
                "border-radius": "10px",
                "margin": "6px 0",
                "padding": "10px 14px",
                "color": "#A0AEC0",
                "transition": "all 0.25s ease",
            },

            "nav-link:hover": {
                "background": "rgba(255,255,255,0.05)",
                "color": "#FFFFFF",
                "transform": "translateX(4px)",
            },

            "nav-link-selected": {
                "background": "rgba(0,201,167,0.15)",
                "color": "#FFFFFF",
                "border": "1px solid rgba(0,201,167,0.4)",
            },
        },
    )


def _stock_settings():
    st.sidebar.header("📈 Stock Settings")
    st.session_state.ticker = st.sidebar.text_input(
        "Stock", value=st.session_state.get("ticker", "AAPL")
    )
    st.session_state.start_date = st.sidebar.date_input(
        "Start date",
        value=st.session_state.get("start_date", date(2015, 1, 1)),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )
    st.session_state.end_date = st.sidebar.date_input(
        "End date",
        value=st.session_state.get("end_date", date.today()),
        min_value=st.session_state.start_date,
        max_value=date.today(),
    )
    if st.sidebar.button("🔄 Reload Data"):
        st.session_state.data = None
        st.session_state.force_reload = True


def sidebar(current_page: str):
    """
    Call once at the top of every page, passing the page's label.
    Stock settings are shown automatically based on the PAGES registry.

    Usage:
        from utils import sidebar
        sidebar("Home")            # no stock controls
        sidebar("Tree")            # stock controls shown
    """
    _hide_default_nav()

    labels = [p["label"] for p in PAGES]

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page

    current_index = labels.index(current_page) if current_page in labels else 0
    page_config = next((p for p in PAGES if p["label"] == current_page), PAGES[0])

    with st.sidebar:
        selection = _nav_menu(current_index)
        if page_config["show_stock_settings"]:
            _stock_settings()

    if selection != current_page:
        st.session_state["current_page"] = selection
        target = next(p for p in PAGES if p["label"] == selection)
        st.switch_page(target["path"])
        
def save_quiz_result(user_id, email, score, answers, quizName):

    ref = db.reference(f"users/{user_id}/quiz_results:{quizName}")

    ref.push({
        "email": email,
        "score": int(score),
        "answers": answers,
        "timestamp": datetime.utcnow().isoformat()
    })

import streamlit as st
from datetime import datetime

def create_quiz(quiz_id, questions, user_id, email):
    
    """
    quiz_id: A unique string for the Firebase path (e.g., 'ohlcv_lab_01')
    questions: A list of dicts containing the question data
    user_id/email: User info for the database record
    db: Your initialized firebase_admin.db objedbct
    """
    
    with st.form(key=f"form_{quiz_id}"):
        st.subheader("🧠 Knowledge Check")
        user_responses = {}

        for i, q in enumerate(questions):
            label = f"{i+1}. {q['question']}"
            
            if q['type'] == 'radio':
                user_responses[q['id']] = st.radio(label, q['options'], key=f"{quiz_id}_{q['id']}")
            
            elif q['type'] == 'text':   
                user_responses[q['id']] = st.text_input(label, key=f"{quiz_id}_{q['id']}")

        submitted = st.form_submit_button("Submit Quiz")

    if submitted:
        # Calculate Score
        score = 0
        total_possible = 0
        
        for q in questions:
            if q['type'] == 'radio':
                total_possible += 1
                if user_responses[q['id']] == q['answer']:
                    score += 1

        # Database Logic (using your existing ref style)
        try:
            ref = db.reference(f"users/{user_id}/quiz_results:{quiz_id}")
            ref.push({
                "email": email,
                "score": score,
                "total_questions": total_possible,
                "answers": user_responses,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            st.success(f"✅ Quiz submitted! Score: {score}/{total_possible}")
            if score == total_possible:
                st.balloons()
        except Exception as e:
            st.error(f"Error saving to database: {e}")
            
def verify_create_quiz(name,lab_specific_questions):
    user_id = st.session_state["user_id"] if "user_id" in st.session_state else None
    email = st.session_state["email"] if "email" in st.session_state else None
    logged_in = st.session_state["logged_in"] if "logged_in" in st.session_state else False

    if logged_in and user_id:
        try:
            create_quiz(
                quiz_id=name, # e.g., "Regression", "Tree", "LSTM"
                questions=lab_specific_questions,
                user_id=user_id,
                email=email
            )
        except Exception as e:
            st.error(f"Error loading quiz: {e}")
    else:
        st.warning("⚠️ **Please log in to take the quiz.** Your progress must be saved to your profile.")

def apply_theme():
    st.markdown("""
    <style>

    /* ===== BASE APP BACKGROUND (DARK + CLEAN) ===== */
    .stApp {
        background-color: #070B10;  /* darker than default */
        color: #FFFFFF;
    }

    /* ===== MAIN CONTAINER ===== */
    .block-container {
        padding: 2rem 3rem;
    }

    /* ===== GLASS CARD SYSTEM ===== */
    div[data-testid="stMetric"],
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
    }

    /* Hover lift */
    div[data-testid="stMetric"]:hover,
    div[data-testid="stDataFrame"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }

    /* Subtle glow for metrics */
    div[data-testid="stMetric"] {
        box-shadow: 0 0 20px rgba(0,201,167,0.06);
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #00C9A7, #4F8CFF);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 8px 20px rgba(0,201,167,0.35);
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #070B10;  /* deeper than main bg */
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* ===== INPUTS ===== */
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #111827;
        color: white;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ===== HEADINGS ===== */
    h1 {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h2 {
        font-size: 1.6rem;
        font-weight: 600;
        color: #E2E8F0;
    }

    h3 {
        color: #A0AEC0;
    }

    /* ===== DIVIDER ===== */
    hr {
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ===== SCROLLBAR (BONUS POLISH) ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0B0F14;
    }

    ::-webkit-scrollbar-thumb {
        background: #1F2937;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #374151;
    }

    </style>
    """, unsafe_allow_html=True)