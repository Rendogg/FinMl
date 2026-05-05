"""
FinML Authentication Module
---------------------------
Handles user login, signup, and session management using Firebase Authentication.
"""

# Standard libs
import json

# Third-party
import streamlit as st
import requests
import firebase_admin
from firebase_admin import credentials

# Local
from utilFunctions import sidebar, apply_theme

# Firebase Setup
firebase_config = dict(st.secrets["firebase"])
cred = credentials.Certificate(firebase_config)


firebase_admin.initialize_app(cred, {
        "databaseURL": "https://finmllab-69af9-default-rtdb.europe-west1.firebasedatabase.app/"
    })

API_KEY = st.secrets["FIREBASE_API_KEY"]
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ===== STYLING =====
st.markdown("""
<style>

/* ===== Static Background ===== */
.stApp {
    background-color: #0E1117;
}

/* ===== Center Layout ===== */
.hero {
    text-align: center;
    margin-top: 15vh;
}

/* ===== Glass Card with Animated Background ===== */
.glass {
    position: relative;
    padding: 50px;
    border-radius: 20px;
    overflow: hidden;
    display: inline-block;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}

/* ===== Animated Layer INSIDE glass ===== */
.glass::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(-45deg, #00C9A7, #4F8CFF, #00C9A7, #1A1F2B);
    background-size: 300% 300%;
    animation: moveGradient 8s ease infinite;
    opacity: 0.25;
    z-index: 0;
}

/* ===== Content stays above animation ===== */
.glass-content {
    position: relative;
    z-index: 1;
}

/* ===== Title ===== */
.title {
    font-size: 3.5rem;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
    animation: slideUp 1s ease forwards;
}

/* ===== Subtitle ===== */
.subtitle {
    color: #A0AEC0;
    font-size: 1.2rem;
    margin-bottom: 30px;
    animation: fadeIn 2s ease-in;
}

/* ===== Button ===== */
.stButton > button {
    background: linear-gradient(135deg, #00C9A7, #4F8CFF);
    color: white;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 25px rgba(0,201,167,0.5);
}

/* ===== Animations ===== */
@keyframes moveGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from {
        transform: translateY(40px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

</style>
""", unsafe_allow_html=True)


# ===== HERO =====
st.markdown("""
<div class="hero">
    <div class="glass">
        <div class="glass-content">
            <div class="title"> Welcome to FinMl </div>
            <div class="subtitle">
                A learning tool for machine learning within finances<br>
                Explore. Train. Predict.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Init sidebar
sidebar("Home")

# Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "email" not in st.session_state:
    st.session_state.email = ""

if "token" not in st.session_state:
    st.session_state.token = ""



def handle_error(data):
    """
    Display Firebase error messages in a user-friendly format.

    Args:
        data (dict): Response dictionary returned from Firebase API.
    """
    if "error" in data:
        message = data["error"]["message"]
        st.error(f"Firebase Error: {message}")


# Firebase REST
def firebase_signup(email, password, username):
    """
    Register a new user using Firebase Authentication.

    Args:
        email (str): User's email address.
        password (str): User's password.
        username (str): Display name for the user.

    Returns:
        dict or None: Firebase response containing user data if successful,
                      otherwise None.
    """
    url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"

    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
        "displayName": username
    }

    r = requests.post(url, params={"key": API_KEY}, data=json.dumps(payload))
    data = r.json()

    if "email" in data:
        return data
    else:
        handle_error(data)


def firebase_login(email, password):
    """
    Authenticate a user with Firebase using email and password.

    Args:
        email (str): User's email address.
        password (str): User's password.

    Returns:
        dict or None: Firebase response containing authentication details
                      (including ID token) if successful, otherwise None.
    """
    url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"

    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    r = requests.post(url, params={"key": API_KEY}, data=json.dumps(payload))
    data = r.json()

    if "localId" in data:
        st.session_state.user_id = data["localId"]

    if "user_id" not in st.session_state:
        st.session_state.user_id = ""

    if "email" in data:
        return data
    else:
        handle_error(data)


def verify_token(token):
    """
    Verify a Firebase ID token and retrieve user information.

    Args:
        token (str): Firebase authentication ID token.

    Returns:
        dict or None: User information if token is valid, otherwise None.
    """
    url = "https://identitytoolkit.googleapis.com/v1/accounts:lookup"

    payload = {"idToken": token}

    r = requests.post(url, params={"key": API_KEY}, data=json.dumps(payload))
    data = r.json()

    if "users" in data:
        return data["users"][0]

    return None


# Login Handler
def login():
    """
    Handle login form submission and update session state.

    Uses credentials stored in Streamlit session state and updates:
    - logged_in status
    - username
    - email
    - authentication token
    """
    data = firebase_login(
        st.session_state.login_email,
        st.session_state.login_password
    )

    if data:
        st.session_state.logged_in = True
        st.session_state.username = data.get("displayName")
        st.session_state.email = data["email"]
        st.session_state.token = data["idToken"]


# Logout
def logout():
    """
    Clear user session and log the user out.
    """
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.token = ""


# Persistent Login Check
if not st.session_state.logged_in and st.session_state.token:

    user = verify_token(st.session_state.token)

    if user:
        st.session_state.logged_in = True
        st.session_state.email = user["email"]
        st.session_state.username = user.get("displayName")

if not st.session_state.logged_in:

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:

        with st.form("login_form"):

            st.text_input("Email", key="login_email")
            st.text_input("Password", type="password", key="login_password")

            login_submit = st.form_submit_button("Login")

            if login_submit:
                with st.spinner("Logging In"):
                    login()

    with signup_tab:

        with st.form("signup_form"):

            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if not email or not password:
                st.warning("Please Fill Out All Fields")

            if len(password) < 6:
                st.warning("Password must be at least 6 characters")

            signup_submit = st.form_submit_button("Create Account")

            if signup_submit:
                user = firebase_signup(email, password, username)
                if user:
                    st.success("Account created successfully")
                    st.balloons()
else:
    st.success(f"Logged in as {st.session_state.username}")
    st.write("Email:", st.session_state.email)

    if st.button("Sign Out"):
        logout()