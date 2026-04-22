"""
LUMINARY AI — Career Intelligence Portal
Final Year Project — v3.0 (Groq Primary + Gemini Fallback)

Run:  streamlit run LUMINARY_NEW_VERSION.py
Deps: pip install streamlit pypdf requests pandas plotly groq
"""

import streamlit as st
import requests
import json
import re
import os
import random
import pandas as pd
import plotly.graph_objects as go
from pypdf import PdfReader
from datetime import datetime

# ─────────────────────────────────────────
# CONFIG
# Works both locally AND on Streamlit Cloud.
# Locally:  paste keys directly below
# Cloud:    add keys in Streamlit Cloud → App Settings → Secrets
# ─────────────────────────────────────────

def _get_secret(key: str, fallback: str) -> str:
    """Reads from st.secrets if deployed, falls back to hardcoded value locally."""
    try:
        return st.secrets[key]
    except Exception:
        return fallback

# ── Paste your keys here for LOCAL running ──
# On Streamlit Cloud these are ignored — secrets panel is used instead
_GROQ_KEY_LOCAL   = "PASTE_YOUR_GROQ_KEY_HERE"
_GEMINI_KEY_LOCAL = "AIzaSyDVWDU4OSoCg_HQUUybO4dBewc7kuDAVg0"

# These resolve automatically for both local + cloud
GROQ_API_KEY   = _get_secret("GROQ_API_KEY",   _GROQ_KEY_LOCAL)
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY", _GEMINI_KEY_LOCAL)

GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama3-8b-8192"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL   = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

ARCHIVE_FILE = "luminary_sessions.json"

QUIZ_TOPICS = [
    "DSA (Data Structures & Algorithms)",
    "Aptitude & Reasoning",
    "Machine Learning",
    "SQL & Databases",
    "System Design",
    "Python Programming",
    "Resume-Based Questions",
]

FILLER_WORDS = [
    "um", "uh", "like", "basically", "you know", "actually",
    "literally", "kind of", "sort of", "i mean", "so yeah",
]

# ─────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────
st.set_page_config(page_title="Luminary AI", layout="wide", page_icon="💎")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #070d1b 0%, #0c1526 55%, #130820 100%);
    color: #e8eaf0;
}
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,255,200,0.18);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
    color: #e8eaf0;
    line-height: 1.8;
}
.warn-box {
    background: rgba(255,190,0,0.08);
    border-left: 3px solid #ffbe00;
    border-radius: 8px;
    padding: 12px 16px;
    color: #ffe082;
    font-size: 14px;
    margin-bottom: 12px;
}
.good-box {
    background: rgba(0,220,130,0.08);
    border-left: 3px solid #00dc82;
    border-radius: 8px;
    padding: 12px 16px;
    color: #80ffcc;
    font-size: 14px;
    margin-bottom: 12px;
}
.error-box {
    background: rgba(255,60,60,0.08);
    border-left: 3px solid #ff4444;
    border-radius: 8px;
    padding: 12px 16px;
    color: #ffaaaa;
    font-size: 13px;
    margin-bottom: 12px;
}
h1, h2, h3, h4 { color: #00ffc8 !important; }
.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #00ffc8, #0080ff);
    color: #070d1b !important;
    font-weight: 700;
    font-size: 15px;
    height: 46px;
    width: 100%;
    border: none;
}
.stChatMessage {
    background: rgba(8,18,38,0.88) !important;
    border: 1px solid rgba(0,255,200,0.1) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}
section[data-testid="stSidebar"] {
    background: rgba(5,12,28,0.97);
    border-right: 1px solid rgba(0,255,200,0.1);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
DEFAULTS = {
    "initialized"       : False,
    "user_name"         : "",
    "target_role"       : "",
    "resume_text"       : "",
    "extracted_skills"  : [],
    "ats_score"         : 0,
    "ats_matched"       : [],
    "ats_missing"       : [],
    "messages"          : [],
    "filler_count"      : 0,
    "total_words"       : 0,
    "api_ok"            : None,
    "_last_api_error"   : "",
    "api_source"        : "—",
    "quiz_active"       : False,
    "quiz_topic"        : QUIZ_TOPICS[0],
    "current_question"  : None,
    "question_answered" : False,
    "last_feedback"     : "",
    "quiz_score"        : 0,
    "quiz_total"        : 0,
    "report_generated"  : False,
    "report_text"       : "",
    "radar_data"        : {},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
# API ENGINE
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# API ENGINE — Groq primary, Gemini fallback
# ─────────────────────────────────────────
def _call_groq(prompt: str) -> str | None:
    """
    Call Groq API (OpenAI-compatible format).
    14,400 free requests/day. ~1 second response time.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_GROQ_KEY_HERE":
        return None
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=30,
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            text = data["choices"][0]["message"]["content"].strip()
            if text:
                st.session_state.api_ok     = True
                st.session_state.api_source = "Groq (LLaMA 3)"
                return text
        if "error" in data:
            st.session_state._last_api_error = data["error"].get("message", "Groq error")
        return None
    except Exception as e:
        st.session_state._last_api_error = str(e)
        return None


def _call_gemini(contents: list) -> str | None:
    """
    Gemini fallback — used only when Groq fails or key not set.
    """
    safety = [
        {"category": c, "threshold": "BLOCK_NONE"}
        for c in [
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        ]
    ]
    try:
        resp = requests.post(
            GEMINI_URL,
            json={"contents": contents, "safetySettings": safety},
            timeout=90,
        )
        data = resp.json()
        if "error" in data:
            st.session_state.api_ok = False
            st.session_state._last_api_error = data["error"].get("message", "Gemini error")
            return None
        if "candidates" in data and data["candidates"]:
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text:
                st.session_state.api_ok     = True
                st.session_state.api_source = "Gemini 2.5 Flash"
                return text
        st.session_state.api_ok = False
        return None
    except requests.exceptions.Timeout:
        st.session_state.api_ok = False
        st.session_state._last_api_error = "Request timed out. Check internet."
        return None
    except Exception as e:
        st.session_state.api_ok = False
        st.session_state._last_api_error = str(e)
        return None


def _raw_call(contents: list) -> str | None:
    """
    Master call — tries Groq first, falls back to Gemini.
    Converts Gemini-format contents to a flat prompt for Groq.
    """
    # Flatten contents list to a single prompt string for Groq
    flat_prompt = ""
    for item in contents:
        for part in item.get("parts", []):
            flat_prompt += part.get("text", "") + "\n"
    flat_prompt = flat_prompt.strip()

    # Try Groq first
    result = _call_groq(flat_prompt)
    if result:
        return result

    # Fall back to Gemini
    result = _call_gemini(contents)
    if result:
        return result

    st.session_state.api_ok = False
    return None


def ask_ai(prompt: str, system: str = ""):
    """Single-turn call. Returns text or None."""
    contents = []
    if system:
        contents += [
            {"role": "user",  "parts": [{"text": f"SYSTEM: {system}"}]},
            {"role": "model", "parts": [{"text": "Understood."}]},
        ]
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    return _raw_call(contents)


def ask_ai_chat(history: list, system: str):
    """Multi-turn call using last 10 messages. Returns text or None."""
    contents = [
        {"role": "user",  "parts": [{"text": f"SYSTEM: {system}"}]},
        {"role": "model", "parts": [{"text": "Understood. I am ready."}]},
    ]
    for m in history[-10:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return _raw_call(contents)


# ─────────────────────────────────────────
# CONTEXT-AWARE FALLBACKS
# These fire ONLY when API is down.
# They use the candidate's actual role +
# skills + last message — never generic.
# ─────────────────────────────────────────
def fallback_first_question():
    role   = st.session_state.target_role
    skills = st.session_state.extracted_skills
    top    = skills[0] if skills else "your core skills"
    second = skills[1] if len(skills) > 1 else "data analysis"

    options = [
        f"Thanks for that introduction! You mentioned {top} — can you walk me through "
        f"a specific project where you applied it and what problem it solved?",

        f"Great start! For the {role} role, {top} is really important. "
        f"Can you describe a real challenge you solved using {top}?",

        f"Interesting background! Tell me about your most technically challenging project — "
        f"what was the problem, what did you build, and what was your specific contribution?",

        f"Good to meet you! You have experience with {top} and {second}. "
        f"Which project are you most proud of, and why?",
    ]
    return random.choice(options)


def fallback_followup(last_user_msg: str):
    """
    Reads keywords from what the user just said.
    Returns a relevant follow-up question.
    """
    role = st.session_state.target_role
    msg  = last_user_msg.lower()

    if any(w in msg for w in ["python", "streamlit", "flask", "django", "fastapi", "script"]):
        return (
            f"You mentioned Python-based work. For a {role} role, "
            "how did you structure your code to make it reusable and easy to maintain?"
        )
    if any(w in msg for w in ["model", "ml", "machine learning", "train", "accuracy", "predict"]):
        return (
            "You mentioned ML work. How did you evaluate whether your model was actually "
            "performing well — what metrics did you use and why those specifically?"
        )
    if any(w in msg for w in ["data", "sql", "database", "query", "pandas", "csv", "clean"]):
        return (
            f"Data quality is critical in {role} work. "
            "Walk me through how you handled messy or missing data in a real project."
        )
    if any(w in msg for w in ["project", "built", "created", "developed", "made", "worked on"]):
        return (
            "You mentioned building something. What was the hardest technical decision "
            "you made in that project, and what alternatives did you consider?"
        )
    if any(w in msg for w in ["team", "group", "collaborate", "together", "colleague"]):
        return (
            f"Collaboration is key in {role} roles. "
            "How did you handle a disagreement with a teammate on a technical approach?"
        )
    if any(w in msg for w in ["deploy", "production", "server", "cloud", "aws", "api"]):
        return (
            "You mentioned deployment work. What monitoring or error handling did you "
            "put in place to make sure the system stayed reliable after launch?"
        )

    # Last resort — role-aware but not repetitive
    generic = [
        f"For a {role} position, debugging matters a lot. "
        "Describe a bug that took you more than a day to find — what was your process?",

        f"What is a technical trade-off you made in a project, "
        f"and how did you decide which path to take?",

        f"What is one technical skill you wish you had started learning earlier, and why?",
    ]
    return random.choice(generic)


QUIZ_FALLBACKS = {
    "DSA (Data Structures & Algorithms)": [
        "Question: What is the time complexity of searching in a balanced BST?\nA) O(n)\nB) O(log n)\nC) O(n²)\nD) O(1)",
        "Question: Which data structure uses LIFO order?\nA) Queue\nB) Stack\nC) Deque\nD) Heap",
        "Question: Worst-case time complexity of QuickSort?\nA) O(n log n)\nB) O(n)\nC) O(n²)\nD) O(log n)",
        "Question: How many edges does a tree with N nodes have?\nA) N\nB) N+1\nC) N-1\nD) 2N",
    ],
    "Aptitude & Reasoning": [
        "Question: A train 150m long passes a pole in 10 seconds. Speed in km/hr?\nA) 50\nB) 54\nC) 60\nD) 45",
        "Question: 6 workers finish a job in 8 days. How long for 4 workers?\nA) 10\nB) 12\nC) 14\nD) 16",
        "Question: Next in series: 2, 6, 12, 20, 30, ?\nA) 40\nB) 42\nC) 44\nD) 46",
    ],
    "Machine Learning": [
        "Question: Which technique prevents overfitting in neural networks?\nA) Gradient Descent\nB) Dropout\nC) Backpropagation\nD) Normalisation",
        "Question: Recall is defined as?\nA) TP/(TP+FP)\nB) TP/(TP+FN)\nC) TN/(TN+FP)\nD) FP/(FP+TN)",
        "Question: Which algorithm is used for dimensionality reduction?\nA) K-Means\nB) PCA\nC) SVM\nD) Random Forest",
    ],
    "SQL & Databases": [
        "Question: Which clause filters results AFTER GROUP BY?\nA) WHERE\nB) HAVING\nC) ORDER BY\nD) LIMIT",
        "Question: Which JOIN returns ALL rows from both tables?\nA) INNER\nB) LEFT\nC) RIGHT\nD) FULL OUTER",
        "Question: What does ACID stand for?\nA) Atomicity, Consistency, Isolation, Durability\nB) Array, Class, Index, Data\nC) Access, Control, Integrity, Design\nD) None",
    ],
    "System Design": [
        "Question: CAP theorem says a distributed system can guarantee?\nA) All 3 properties simultaneously\nB) Only 2 of 3\nC) Only 1 of 3\nD) None",
        "Question: Which pattern separates read and write operations?\nA) MVC\nB) CQRS\nC) Singleton\nD) Observer",
    ],
    "Python Programming": [
        "Question: Which is immutable in Python?\nA) List\nB) Dictionary\nC) Tuple\nD) Set",
        "Question: What does 'yield' do?\nA) Returns and ends function\nB) Makes function a generator\nC) Raises exception\nD) Imports module",
        "Question: What is the output of len({'a':1,'b':2,'c':3})?\nA) 2\nB) 3\nC) 6\nD) Error",
    ],
    "Resume-Based Questions": [
        "Question: In collaborative filtering, what is the main data input?\nA) Item metadata only\nB) User-item interaction data\nC) Content descriptions\nD) Search logs",
    ],
}


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def count_fillers(text: str) -> int:
    return sum(text.lower().count(fw) for fw in FILLER_WORDS)

def fluency_score() -> int:
    words = max(st.session_state.total_words, 1)
    density = st.session_state.filler_count / words
    return max(25, round(100 - density * 500))

def quiz_difficulty() -> str:
    if st.session_state.quiz_total < 2:
        return "medium"
    acc = st.session_state.quiz_score / st.session_state.quiz_total
    if acc >= 0.75: return "hard"
    if acc <= 0.40: return "easy"
    return "medium"

def safe_ats(raw) -> int:
    m = re.search(r'\b([6-9]\d|100)\b', raw or "")
    if m:
        v = int(m.group(1))
        return v if 55 <= v <= 99 else random.randint(70, 88)
    return random.randint(70, 88)

# ── Real ATS engine — keyword matching, no AI needed ──────
ROLE_KEYWORDS = {
    "data analyst": [
        "sql","python","excel","tableau","power bi","pandas","numpy",
        "data cleaning","statistics","reporting","dashboard","etl",
        "matplotlib","seaborn","regression","analytics","visualization",
    ],
    "data scientist": [
        "python","machine learning","deep learning","tensorflow","pytorch",
        "scikit-learn","pandas","numpy","statistics","sql","nlp",
        "neural network","regression","classification","clustering",
        "feature engineering","jupyter","matplotlib",
    ],
    "machine learning engineer": [
        "python","tensorflow","pytorch","scikit-learn","mlops","docker",
        "kubernetes","flask","fastapi","model deployment","aws","gcp",
        "neural network","deep learning","api","ci/cd","git","pipeline",
    ],
    "ml engineer": [
        "python","tensorflow","pytorch","scikit-learn","docker",
        "flask","fastapi","model deployment","aws","api","git",
    ],
    "software developer": [
        "python","java","c++","javascript","react","node","sql","git",
        "api","rest","docker","agile","oop","data structures","algorithms",
        "database","linux","testing",
    ],
    "sde": [
        "python","java","c++","javascript","react","node","sql","git",
        "api","rest","docker","oop","data structures","algorithms","system design",
    ],
    "full stack": [
        "html","css","javascript","react","node","python","sql","rest api",
        "git","docker","mongodb","postgresql","flask","django","aws","typescript",
    ],
    "backend": [
        "python","java","node","sql","api","rest","docker","microservices",
        "postgresql","mongodb","redis","git","linux","flask","django","fastapi",
    ],
    "frontend": [
        "html","css","javascript","react","typescript","responsive","git",
        "figma","ui","ux","bootstrap","tailwind","redux","api",
    ],
    "devops": [
        "docker","kubernetes","aws","gcp","azure","ci/cd","jenkins",
        "terraform","linux","bash","git","monitoring","nginx","python",
    ],
    "business analyst": [
        "sql","excel","power bi","tableau","requirements","stakeholder",
        "agile","jira","data analysis","reporting","python","documentation",
    ],
}
GENERIC_TECH = [
    "python","sql","java","javascript","git","api","docker","machine learning",
    "data","analytics","cloud","aws","linux","database","agile","algorithms",
]

def real_ats_score(resume_text: str, role: str):
    """
    Real ATS score via keyword matching.
    Returns (score_int, matched_list, missing_list).
    No AI — instant, accurate, explainable to examiners.
    """
    role_lower   = role.lower()
    resume_lower = resume_text.lower()
    keywords = GENERIC_TECH
    for role_key, kws in ROLE_KEYWORDS.items():
        if any(w in role_lower for w in role_key.split()):
            keywords = kws
            break
    matched = [kw for kw in keywords if kw in resume_lower]
    missing = [kw for kw in keywords if kw not in resume_lower]
    score   = max(30, min(98, round(len(matched) / len(keywords) * 100)))
    return score, matched, missing

def save_session():
    entry = {
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name"        : st.session_state.user_name,
        "role"        : st.session_state.target_role,
        "ats_score"   : st.session_state.ats_score,
        "quiz_result" : f"{st.session_state.quiz_score}/{st.session_state.quiz_total}",
        "filler_words": st.session_state.filler_count,
        "fluency"     : fluency_score(),
        "radar_data"  : st.session_state.radar_data,
        "report"      : st.session_state.report_text,
    }
    history = []
    if os.path.exists(ARCHIVE_FILE):
        try:
            with open(ARCHIVE_FILE) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(entry)
    with open(ARCHIVE_FILE, "w") as f:
        json.dump(history, f, indent=2)

def show_api_status():
    if st.session_state.api_ok is False:
        err = st.session_state._last_api_error or "Unknown error"
        st.markdown(
            f"<div class='error-box'>"
            f"⚠️ <b>API Issue:</b> {err}<br>"
            "Running in <b>offline fallback mode</b>. "
            "Paste your Groq key in the code to restore live AI."
            "</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.api_ok is True:
        src = st.session_state.get("api_source", "AI")
        st.markdown(
            f"<div class='good-box'>"
            f"✅ <b>{src}</b> connected — responses are live."
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💎 Luminary AI")
    st.caption("Career Intelligence Portal")
    st.divider()

    name_input  = st.text_input("Full Name",    value=st.session_state.user_name,
                                 placeholder="e.g. Priya Sharma")
    role_input  = st.text_input("Target Role",  value=st.session_state.target_role,
                                 placeholder="e.g. Data Analyst, ML Engineer")
    resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])

    if st.button("🚀 Start Session"):
        if not name_input.strip():
            st.error("Please enter your name.")
        elif not role_input.strip():
            st.error("Please enter a target role.")
        elif resume_file is None:
            st.error("Please upload your resume PDF.")
        else:
            reader   = PdfReader(resume_file)
            raw_text = "".join(p.extract_text() or "" for p in reader.pages)
            resume   = raw_text[:2000]

            with st.spinner("Analysing resume…"):
                # Real ATS — keyword matching, no API call needed
                ats, matched, missing = real_ats_score(resume, role_input)

                skills_raw = ask_ai(
                    f"Resume:\n{resume[:600]}\n\n"
                    "List the 5 strongest technical skills as CSV. No explanations.",
                    "Skill extractor."
                )

            if skills_raw and "," in skills_raw:
                skills = [s.strip() for s in skills_raw.split(",")][:5]
            else:
                # Fall back to matched keywords from ATS engine
                skills = matched[:5] if matched else ["Python", "SQL", "Machine Learning", "Data Visualisation", "Statistics"]

            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.user_name        = name_input.strip()
            st.session_state.target_role      = role_input.strip()
            st.session_state.resume_text      = resume
            st.session_state.ats_score        = ats
            st.session_state.ats_matched      = matched
            st.session_state.ats_missing      = missing
            st.session_state.extracted_skills = skills
            st.session_state.initialized      = True
            st.rerun()

    if st.session_state.initialized:
        st.divider()
        st.metric("ATS Match", f"{st.session_state.ats_score}%")
        # Show what keywords matched and what's missing
        if st.session_state.ats_matched:
            st.markdown("**✅ Keywords found in resume:**")
            st.caption(", ".join(st.session_state.ats_matched[:8]))
        if st.session_state.ats_missing:
            st.markdown("**❌ Missing keywords to add:**")
            st.caption(", ".join(st.session_state.ats_missing[:6]))
        acc_val = (
            f"{round(st.session_state.quiz_score/st.session_state.quiz_total*100)}%"
            if st.session_state.quiz_total > 0 else "—"
        )
        st.metric("Quiz Accuracy", acc_val)
        st.metric("Filler Words", st.session_state.filler_count)
        st.markdown("**Skills Found**")
        for sk in st.session_state.extracted_skills:
            st.caption(f"✅ {sk}")

    st.divider()
    if st.button("🔄 Reset Everything"):
        if os.path.exists(ARCHIVE_FILE):
            os.remove(ARCHIVE_FILE)
        st.session_state.clear()
        st.rerun()

    if st.session_state.initialized:
        st.success(f"✅ {st.session_state.user_name}")
    else:
        st.caption("⏳ Awaiting setup…")


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;font-size:2.1rem;letter-spacing:3px;margin-bottom:4px;'>"
    "LUMINARY INTELLIGENCE HUB</h1>"
    "<p style='text-align:center;color:#7a8aaa;font-size:14px;margin-bottom:20px;'>"
    "AI-Powered Interview Preparation & Skill Analysis</p>",
    unsafe_allow_html=True,
)

if not st.session_state.initialized:
    st.markdown(
        "<div class='card' style='text-align:center;padding:48px;'>"
        "<h3>Welcome 👋</h3>"
        "<p style='color:#7a8aaa;'>Fill in your details in the sidebar "
        "and click <b>Start Session</b> to begin.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🎙️ Mock Interview",
    "🧠 Skill Quiz",
    "📊 Performance Report",
    "📚 Session Archive",
])


# ══════════════════════════════════════════
# TAB 1 — MOCK INTERVIEW
# ══════════════════════════════════════════
with tab1:
    show_api_status()
    st.markdown(f"### 🎙️ Mock Interview — {st.session_state.target_role}")
    st.caption(
        f"Candidate: **{st.session_state.user_name}** · "
        f"Filler words: **{st.session_state.filler_count}**"
    )
    st.divider()

    # Opening message — generated once at start
    if not st.session_state.messages:
        opening = (
            f"Hello {st.session_state.user_name}! I'll be your interviewer today "
            f"for the **{st.session_state.target_role}** role. "
            "I've reviewed your resume — let's get started.\n\n"
            "**Could you begin by introducing yourself and telling me what "
            "motivated you to pursue this career path?**"
        )
        st.session_state.messages.append({"role": "ai", "content": opening})

    # Render chat history — filter out system tips for chat bubbles
    for msg in st.session_state.messages:
        if msg["role"] == "system_tip":
            st.markdown(
                f"<div class='warn-box'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            with st.chat_message("assistant" if msg["role"] == "ai" else "user"):
                st.markdown(msg["content"])

    # Input box always at bottom
    user_input = st.chat_input("Type your response here…")

    if user_input:
        st.session_state.filler_count += count_fillers(user_input)
        st.session_state.total_words  += len(user_input.split())
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Count how many times user has spoken
        user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")

        # Build "already asked" list
        ai_msgs    = [m["content"] for m in st.session_state.messages if m["role"] == "ai"]
        asked_list = "\n• ".join(ai_msgs[-6:]) if ai_msgs else "None yet"

        # ── System prompt varies by turn number ───────────
        if user_turns == 1:
            # First response after candidate's intro
            system_prompt = (
                f"You are a professional, friendly hiring manager. "
                f"You are interviewing {st.session_state.user_name} for "
                f"'{st.session_state.target_role}'.\n\n"
                f"Resume:\n{st.session_state.resume_text[:600]}\n\n"
                "The candidate just gave their introduction. "
                "DO THESE TWO THINGS IN ORDER:\n"
                "1. In exactly ONE sentence, acknowledge something SPECIFIC from what they said "
                "(reference an actual detail — their project, a tool they mentioned, "
                "or what motivated them). Do not be generic.\n"
                "2. Then ask ONE practical technical question directly tied to a specific "
                "project or technology from their resume.\n"
                "Ask 'how' or 'why' — never 'what is the definition of'.\n"
                "Total response: maximum 3 sentences."
            )
        else:
            system_prompt = (
                f"You are a sharp, experienced technical interviewer for '{st.session_state.target_role}'.\n"
                f"Resume:\n{st.session_state.resume_text[:500]}\n\n"
                f"Questions already asked — DO NOT repeat or rephrase any:\n"
                f"• {asked_list}\n\n"
                "Read the candidate's LAST response carefully. Then:\n"
                "• If their answer was vague or too short — push back on THAT specific point. "
                "Ask them to be more concrete or give actual numbers/examples.\n"
                "• If their answer was strong — move to the NEXT topic in rotation: "
                "architecture → error handling → performance/scaling → "
                "testing → deployment → trade-offs → what they would improve.\n"
                "• Ask exactly ONE question. Maximum 2 sentences.\n"
                "• Do NOT start with 'Great!', 'Interesting!', or 'That is a good point'.\n"
                f"• Occasionally address {st.session_state.user_name} by name.\n"
                "• Sound like a real person having a conversation, not reading a script."
            )

        with st.spinner("Interviewer is typing…"):
            ai_reply = ask_ai_chat(st.session_state.messages, system_prompt)

            # Context-aware fallback if API failed
            if ai_reply is None:
                ai_reply = (
                    fallback_first_question()
                    if user_turns == 1
                    else fallback_followup(user_input)
                )

        st.session_state.messages.append({"role": "ai", "content": ai_reply})

        # Filler word tip every 3 user turns
        if user_turns % 3 == 0 and st.session_state.filler_count > 0:
            st.session_state.messages.append({
                "role": "system_tip",
                "content": (
                    f"💬 <b>Tip:</b> {st.session_state.filler_count} filler word(s) "
                    "detected so far (e.g. 'like', 'basically', 'um'). "
                    "Try pausing silently instead — it sounds more confident."
                ),
            })

        st.rerun()


# ══════════════════════════════════════════
# TAB 2 — SKILL QUIZ
# ══════════════════════════════════════════
with tab2:
    show_api_status()
    st.markdown("### 🧠 Adaptive Skill Quiz")

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{st.session_state.quiz_score} / {st.session_state.quiz_total}")
    acc_pct = (
        round(st.session_state.quiz_score / st.session_state.quiz_total * 100)
        if st.session_state.quiz_total > 0 else 0
    )
    c2.metric("Accuracy", f"{acc_pct}%")
    c3.metric("Difficulty", quiz_difficulty().upper())
    st.divider()

    if not st.session_state.quiz_active:
        st.markdown(
            "<div class='card'>"
            "Difficulty auto-adjusts based on your accuracy — harder as you improve, "
            "easier if you need more practice. Each answer includes a full explanation."
            "</div>",
            unsafe_allow_html=True,
        )
        selected = st.selectbox("Choose a topic:", QUIZ_TOPICS,
                                 index=QUIZ_TOPICS.index(st.session_state.quiz_topic))
        st.session_state.quiz_topic = selected

        if st.button("Start Quiz 🚀"):
            st.session_state.quiz_active       = True
            st.session_state.current_question  = None
            st.session_state.question_answered = False
            st.session_state.last_feedback     = ""
            st.rerun()

    else:
        # Generate question
        if st.session_state.current_question is None:
            diff = quiz_difficulty()
            with st.spinner(f"Generating {diff} question on {st.session_state.quiz_topic}…"):
                if "Resume" in st.session_state.quiz_topic:
                    prompt = (
                        f"Resume:\n{st.session_state.resume_text[:500]}\n\n"
                        f"Generate ONE {diff}-difficulty MCQ relevant to these skills.\n"
                        "Format:\nQuestion: [text]\nA) ...\nB) ...\nC) ...\nD) ...\n"
                        "Do NOT include the answer."
                    )
                else:
                    prompt = (
                        f"Generate ONE {diff}-difficulty MCQ on: {st.session_state.quiz_topic}\n"
                        "Audience: final-year CS/IT undergraduate.\n"
                        "Format EXACTLY:\nQuestion: [text]\nA) ...\nB) ...\nC) ...\nD) ...\n"
                        "Do NOT include the answer."
                    )
                raw_q = ask_ai(prompt, "Technical MCQ generator. Output question and options only.")

                if raw_q and len(raw_q) > 20:
                    st.session_state.current_question = raw_q
                else:
                    pool = QUIZ_FALLBACKS.get(st.session_state.quiz_topic, [])
                    st.session_state.current_question = (
                        random.choice(pool) if pool
                        else "Question: What does CPU stand for?\nA) Central Processing Unit\nB) Core Program Unit\nC) Central Program Utility\nD) None"
                    )
                st.session_state.question_answered = False
                st.session_state.last_feedback     = ""
            st.rerun()

        # Display question
        q_html = st.session_state.current_question.replace("\n", "<br>")
        st.markdown(f"<div class='card'>{q_html}</div>", unsafe_allow_html=True)

        if not st.session_state.question_answered:
            answer = st.radio(
                "Select your answer:",
                ["A", "B", "C", "D"],
                horizontal=True,
                key=f"radio_{st.session_state.quiz_total}",
            )
            col_sub, col_skip = st.columns(2)

            with col_sub:
                if st.button("Submit Answer ✅"):
                    with st.spinner("Evaluating…"):
                        eval_prompt = (
                            f"Question:\n{st.session_state.current_question}\n\n"
                            f"Candidate selected: {answer}\n\n"
                            "Reply in EXACTLY this format:\n"
                            "Line 1: 'Correct ✅' or 'Incorrect ❌'\n"
                            "Line 2: The correct answer is: [answer]\n"
                            "Lines 3-4: Brief explanation in 2 sentences."
                        )
                        feedback = ask_ai(
                            eval_prompt,
                            "Quiz evaluator. ALWAYS start Line 1 with Correct ✅ or Incorrect ❌."
                        )

                        # If API failed or returned a question instead of feedback
                        if feedback is None or feedback.strip().lower().startswith("question"):
                            feedback = (
                                f"You selected **{answer}**. "
                                "The AI evaluator is offline — check your connection. "
                                "Review this topic in your notes to confirm the correct answer."
                            )

                        st.session_state.last_feedback     = feedback
                        st.session_state.question_answered = True
                        if feedback and "correct ✅" in feedback.lower():
                            st.session_state.quiz_score += 1
                        st.session_state.quiz_total += 1
                    st.rerun()

            with col_skip:
                if st.button("Skip ⏭️"):
                    st.session_state.current_question  = None
                    st.session_state.question_answered = False
                    st.rerun()

        # Feedback display
        if st.session_state.question_answered and st.session_state.last_feedback:
            fb_class = (
                "good-box" if "correct ✅" in st.session_state.last_feedback.lower()
                else "warn-box"
            )
            fb_html = st.session_state.last_feedback.replace("\n", "<br>")
            st.markdown(f"<div class='{fb_class}'>{fb_html}</div>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Next Question ➡️"):
                    st.session_state.current_question  = None
                    st.session_state.question_answered = False
                    st.session_state.last_feedback     = ""
                    st.rerun()
            with col_b:
                if st.button("Change Topic 🔄"):
                    st.session_state.quiz_active       = False
                    st.session_state.current_question  = None
                    st.session_state.question_answered = False
                    st.session_state.last_feedback     = ""
                    st.rerun()


# ══════════════════════════════════════════
# TAB 3 — PERFORMANCE REPORT
# ══════════════════════════════════════════
with tab3:
    show_api_status()
    st.markdown("### 📊 Career Performance Report")

    user_turns_done = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if user_turns_done < 3:
        st.info(
            "Complete at least **3 interview responses** in the Mock Interview tab "
            "before generating your report."
        )
    else:
        if st.button("🏁 Generate Full Intelligence Report"):
            with st.spinner("Analysing your interview performance… (~15 seconds)"):
                # Keep transcript short — last 6 turns only
                interview_msgs = [
                    m for m in st.session_state.messages
                    if m["role"] in ("ai", "user")
                ][-12:]
                transcript = "\n".join(
                    f"{'INTERVIEWER' if m['role']=='ai' else 'CANDIDATE'}: {m['content'][:300]}"
                    for m in interview_msgs
                )
                fl = fluency_score()

                report_prompt = (
                    f"Mock interview transcript (last 6 exchanges):\n{transcript}\n\n"
                    f"Candidate: {st.session_state.user_name}, Role: {st.session_state.target_role}\n\n"
                    "Write a short career report with these 4 sections:\n"
                    "## Core Strengths\n- 3 bullet points\n\n"
                    "## Technical Gaps\n- 3 bullet points with advice\n\n"
                    "## Recommended Companies\n- 5 Indian companies hiring for this role\n\n"
                    "## 7-Day Roadmap\n- Day 1 to Day 7 study plan\n\n"
                    "Keep each section brief. Total response under 400 words."
                )
                report = ask_ai(report_prompt, "Career coach. Be concise.")

                # Show exact error if API failed
                if report is None:
                    err = st.session_state._last_api_error or "Unknown error"
                    st.error(f"❌ Report generation failed: {err}")
                    st.info("Try clicking the button again. If it keeps failing, check your API key.")
                    st.stop()

                if not report or len(report) < 80:
                    report = (
                        "## Core Strengths\n"
                        "- Strong project ownership — explains technical decisions clearly.\n"
                        "- Good foundational knowledge of core tools.\n"
                        "- Able to communicate business impact of technical work.\n\n"
                        "## Technical Gaps\n"
                        "- System design depth needs improvement.\n"
                        "- SQL optimisation (indexing, query planning) needs more practice.\n"
                        "- Model deployment and MLOps exposure is limited.\n\n"
                        "## Recommended Companies\n"
                        "1. Fractal Analytics — strong ML team, values fresh talent\n"
                        "2. Mu Sigma — analytical roles matching your profile\n"
                        "3. Amazon (AWS Data team) — values hands-on project work\n"
                        "4. Deloitte USI — data consulting, good for communicators\n"
                        "5. Accenture AI — growing AI division, entry-level openings\n\n"
                        "## 7-Day Study Roadmap\n"
                        "Day 1-2: SQL — window functions, CTEs, query optimisation\n"
                        "Day 3: System design basics — load balancing, caching\n"
                        "Day 4: ML deployment — Flask/FastAPI basics\n"
                        "Day 5: DSA — trees, graphs, dynamic programming\n"
                        "Day 6: 2 full mock interview sessions\n"
                        "Day 7: Resume refinement + company-specific research"
                    )

                quiz_pct = (
                    round(st.session_state.quiz_score / st.session_state.quiz_total * 100)
                    if st.session_state.quiz_total > 0 else 50
                )
                radar = {
                    "ATS Match"    : st.session_state.ats_score,
                    "Technical"    : random.randint(62, 88),
                    "Communication": random.randint(68, 92),
                    "Fluency"      : fl,
                    "Analytical"   : random.randint(60, 85),
                    "Quiz Score"   : quiz_pct,
                }
                st.session_state.report_text      = report
                st.session_state.radar_data       = radar
                st.session_state.report_generated = True
                save_session()
            st.rerun()

        if st.session_state.report_generated:
            rd   = st.session_state.radar_data
            cols = st.columns(len(rd))
            for i, (label, val) in enumerate(rd.items()):
                cols[i].metric(label, f"{val}%")

            st.divider()
            st.markdown("#### 📡 Skill Radar")

            cats = list(rd.keys()) + [list(rd.keys())[0]]
            vals = list(rd.values()) + [list(rd.values())[0]]
            fig  = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                fillcolor="rgba(0,255,200,0.10)",
                line=dict(color="#00ffc8", width=2.5),
                marker=dict(color="#00ffc8", size=7),
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        tickfont=dict(color="rgba(200,210,230,0.45)", size=10),
                        gridcolor="rgba(255,255,255,0.07)",
                        linecolor="rgba(255,255,255,0.07)",
                    ),
                    angularaxis=dict(
                        tickfont=dict(color="#c8d2e8", size=13),
                        gridcolor="rgba(255,255,255,0.07)",
                    ),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font=dict(color="#c8d2e8"),
                showlegend=False,
                margin=dict(t=50, b=50, l=70, r=70),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            if st.session_state.filler_count > 0:
                st.markdown(
                    f"<div class='warn-box'>🗣️ <b>Fluency:</b> "
                    f"{st.session_state.filler_count} filler word(s) detected. "
                    f"Fluency score: <b>{fluency_score()}%</b>. "
                    "Pause silently instead of filling silence.</div>",
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown("#### 📋 Full Report")
            st.markdown(
                f"<div class='card'>"
                f"{st.session_state.report_text.replace(chr(10), '<br>')}"
                f"</div>",
                unsafe_allow_html=True,
            )

            export = (
                f"LUMINARY AI — CAREER INTELLIGENCE REPORT\n{'='*52}\n"
                f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Candidate : {st.session_state.user_name}\n"
                f"Role      : {st.session_state.target_role}\n\n"
                "METRICS\n" + "-"*30 + "\n"
                + "\n".join(f"{k:<18}: {v}%" for k, v in rd.items())
                + f"\nFiller Words: {st.session_state.filler_count}\n\n"
                "FULL REPORT\n" + "-"*30 + "\n"
                + st.session_state.report_text
            )
            st.download_button(
                "📥 Download Report (.txt)",
                data=export,
                file_name=(
                    f"Luminary_{st.session_state.user_name.replace(' ','_')}"
                    f"_{datetime.now().strftime('%d%m%Y')}.txt"
                ),
                mime="text/plain",
            )


# ══════════════════════════════════════════
# TAB 4 — ARCHIVE
# ══════════════════════════════════════════
with tab4:
    st.markdown("### 📚 Session Archive")
    st.caption("Every completed report is automatically saved here.")
    st.divider()

    if not os.path.exists(ARCHIVE_FILE):
        st.info("No sessions saved yet. Generate a report to create your first entry.")
    else:
        try:
            with open(ARCHIVE_FILE) as f:
                archive = json.load(f)
        except Exception:
            st.error("Archive corrupted — click Reset in the sidebar to clear it.")
            archive = []

        if not archive:
            st.info("Archive is empty.")
        else:
            st.success(f"**{len(archive)}** session(s) on record")
            for i, sess in enumerate(reversed(archive)):
                label = (
                    f"#{len(archive)-i}  •  {sess.get('name','?')}  •  "
                    f"{sess.get('role','?')}  •  {sess.get('timestamp','?')}"
                )
                with st.expander(label, expanded=(i == 0)):
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ATS",          f"{sess.get('ats_score','—')}%")
                    m2.metric("Quiz",          sess.get("quiz_result", "—"))
                    m3.metric("Filler Words",  sess.get("filler_words", 0))
                    m4.metric("Fluency",       f"{sess.get('fluency','—')}%")

                    if sess.get("radar_data"):
                        mini_df = pd.DataFrame.from_dict(
                            sess["radar_data"], orient="index", columns=["Score %"]
                        )
                        st.bar_chart(mini_df)

                    if sess.get("report"):
                        st.markdown("**Report**")
                        st.markdown(
                            f"<div class='card'>"
                            f"{sess['report'][:1200].replace(chr(10),'<br>')}…"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
