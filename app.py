import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# إعداد الصفحة
# ============================================================
st.set_page_config(
    page_title="محلل الأنماط الإحصائي",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body { direction: rtl; font-family: 'Segoe UI', sans-serif; }

    .card {
        background: linear-gradient(135deg, #0d1b2a, #1a3a5c);
        border-radius: 14px;
        padding: 22px;
        margin: 8px 0;
        border: 1px solid #1e4a7a;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .card-title {
        color: #87ceeb;
        font-size: 0.85em;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .card-value {
        color: #00d4ff;
        font-size: 2em;
        font-weight: bold;
    }
    .card-sub {
        color: #4a7a9b;
        font-size: 0.8em;
        margin-top: 4px;
    }

    .result-pass {
        background: rgba(0,255,136,0.08);
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
        color: #00ff88;
    }
    .result-fail {
        background: rgba(255,68,68,0.08);
        border: 1px solid #ff4444;
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
        color: #ff4444;
    }
    .result-warn {
        background: rgba(255,165,0,0.08);
        border: 1px solid #ffa500;
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
        color: #ffa500;
    }

    .kelly-box {
        background: linear-gradient(135deg, #0a2a1a, #0d4a2a);
        border: 2px solid #00ff88;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .kelly-value {
        font-size: 2.8em;
        font-weight: bold;
        color: #00ff88;
    }

    .rec-strong {
        background: rgba(0,255,136,0.10);
        border-left: 4px solid #00ff88;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #00ff88;
    }
    .rec-warn {
        background: rgba(255,165,0,0.10);
        border-left: 4px solid #ffa500;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #ffa500;
    }
    .rec-danger {
        background: rgba(255,68,68,0.10);
        border-left: 4px solid #ff4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #ff4444;
    }

    .section-title {
        color: #00d4ff;
        font-size: 1.4em;
        font-weight: bold;
        margin: 20px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #1a3a5c;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070e1a, #0d1f35);
    }

    .stTabs [data-baseweb="tab"] {
        color: #87ceeb;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff;
        border-bottom-color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# البيانات الافتراضية
# ============================================================
DEFAULT_DATA = [
    8.72, 6.75, 1.86, 2.18, 1.25, 2.28, 1.24, 1.2, 1.54, 24.46, 4.16, 1.49,
    1.09, 1.47, 1.54, 1.53, 2.1, 32.04, 11, 1.17, 1.7, 2.61, 1.26, 22.23,
    1.77, 1.93, 3.35, 7.01, 1.83, 9.39, 3.31, 2.04, 1.3, 6.65, 1.16, 3.39,
    1.95, 10.85, 1.65, 1.22, 1.6, 4.67, 1.85, 2.72, 1, 3.02, 1.35, 1.3,
    1.37, 17.54, 1.18, 1, 14.4, 1.11, 6.15, 2.39, 2.22, 1.42, 1.23, 2.42,
    1.07, 1.24, 2.55, 7.26, 1.69, 5.1, 2.59, 5.51, 2.31, 2.12, 1.97, 1.5,
    3.01, 2.29, 1.36, 4.95, 5.09, 8.5, 1.77, 5.52, 3.93, 1.5, 2.28, 2.49,
    18.25, 1.68, 1.42, 2.12, 4.17, 1.04, 2.35, 1, 1.01, 5.46, 1.13, 2.84,
    3.39, 2.79, 1.59, 1.53, 4.34, 2.96, 1.06, 1.72, 2.16, 2.2, 3.61, 2.34,
    4.49, 1.72, 1.78, 9.27, 8.49, 2.86, 1.66, 4.63, 9.25, 1.35, 1, 1.64,
    1.86, 2.81, 2.44, 1.74, 1.1, 1.29, 1.45, 8.92, 1.24, 6.39, 1.16, 1.19,
    2.4, 4.64, 3.17, 24.21, 1.17, 1.42, 2.13, 1.12, 3.78, 1.12, 1.52, 22.81,
    1.31, 1.9, 1.38, 1.47, 2.86, 1.79, 1.49, 1.38, 1.84, 1.06, 3.3, 5.97,
    1, 2.92, 1.64, 5.32, 3.26, 1.78, 2.24, 3.16, 1.6, 1.08, 1.55, 1.07,
    1.02, 1.23, 1.08, 5.22, 3.32, 24.86, 3.37, 5.16, 1.69, 2.31, 1.07, 1.1,
    1.01, 1.36, 1.38, 1.54, 5.34, 2.68, 5.78, 3.63, 1.89, 8.41, 4.06, 1.44,
    1.5, 3.17, 1.02, 1.8, 1.9, 1.86, 1.85, 1.73, 3.86, 3.11, 2.44, 1.15,
    2.03, 1.05, 3.05, 1.88, 10.13, 2.29, 1.41, 1, 5.46, 1.26, 23.33, 1.96,
    1.03, 4.54, 1.37, 3.5, 1.13, 1.16, 1.43, 1.13, 1.05, 33.27, 9.96, 1.79,
    2.07, 18.51, 5.75, 1.15, 1.08, 5.92, 1.38, 1.61, 12.99, 24.72, 4.86,
    1.11, 2.86, 1.54, 3.71, 4, 7.57, 2.03, 2.18, 5.52, 13.37, 3.73, 2.41,
    1.79, 5.57, 4.36, 12.33, 1.61, 3.28, 2.89, 1.47, 1.08, 26.89, 1.53,
    2.94, 5.29, 1.23, 1.57, 1.12, 5.69, 3.29, 2.72, 1.18, 5.03, 1.1, 1.32,
    1.18, 1.07, 1.27, 4.6, 11.68, 1.74, 3.94, 3.63, 1.05, 1.61, 1.62, 2.41,
    6.9, 2.02, 1.01, 3.22, 17.21, 1.95, 8.8, 1.44, 2.76, 3.1, 2.84, 1.35,
    1.84, 1.6, 10.72, 1.17, 3.47, 1.45, 1.29, 1.46, 2.23, 12.3, 3.27, 1.23,
    1.02, 1.66, 3.79, 2.06, 4.55, 7.95, 8.55, 4.08, 2.02, 1.21, 1.19, 1.53,
    4.9, 1.84, 10.51, 1.01, 1.34, 1.5, 1.4, 1.42, 4.18, 7.99, 1.23, 1.67,
    3.16, 1.64, 25.06, 4.52, 1.5, 3.23, 1.09, 1.45, 2.77, 7.42, 7.48, 1.89,
    2.11, 4.1, 1.26, 2.29, 10.12, 1.35, 13.21, 2.36, 22.35, 1.76, 2.22,
    1.04, 1.18, 3.69, 1.47, 10.2, 1.47, 1.68, 2.45, 1.03, 2.04, 1.47, 1.18,
    1.72, 1, 3.25, 1.1, 8.74, 1.01, 1.54, 1.34, 5.22, 5.31, 4.47, 2.78,
    21.37, 3.38, 1.63, 2.21, 2.35, 2.14, 1.46, 1.25, 1.67, 1.08, 3.94, 1.66,
    31.1, 1.73, 2.18, 2.06, 1.08, 1.11, 1, 1.07, 1.31, 1.55, 1.98, 1.75,
    1.23, 1.32, 2.56, 3.21, 1.81, 2.09, 1.34, 3.42, 1.29, 1.36, 1.76, 1.61,
    4.52, 1.08, 1.97, 3.75, 1.8, 6.36, 1.14, 1.72, 2.39, 1.28, 4.22, 2.12,
    1.28, 1.38, 1.42, 28.26, 2.15, 1.31, 1.65, 2.43, 2.76, 1.54, 1.61,
    11.91, 2.93, 8.1, 2.04, 1.84, 1.26, 3.69, 3.97, 3.01, 3.16, 1.3, 7.9,
    1.72, 5.57, 2.42, 1.74, 2.06, 2.86, 1.56, 1.4, 2.35, 2.82, 4.03, 1.28,
    2.21, 1.1, 2.06, 1.14, 1.58, 27.78, 2.04, 1.52, 1.22, 1.4, 1.29, 1.16,
    11.72, 1.33, 1.3, 4.34, 1.02, 1.63, 1.9, 9, 1.42, 3.13, 3.8, 1.02,
    1.25, 2.45, 1.74, 1.06, 1.38, 3.46, 1.08, 1, 1.02, 1.84, 1, 1.77, 3.07,
    5.26, 1.73, 1.07, 3.75, 2.32, 1.6, 1.22, 1.72, 2.01, 1.11, 2.03, 1.17,
    1.98, 2.18, 34.49, 1.2, 10.3, 3.4, 2.58, 2.2, 3.16, 29.22, 4.26, 3.18,
    3.29, 1.09, 2.3, 1.25, 3.05, 2.99, 2.16, 3.02, 2.21, 1.59, 5.74, 1.02,
    1.12, 1.21, 2.25, 4.38, 1.05, 1.05, 1.9, 23.03, 4.93, 1.03, 16.7, 4.08,
    1.68, 2.4, 2.89, 2.85, 2.75, 20.29, 3.57, 9.68, 1.46, 5.73, 4.84, 1.15,
    1.92, 3.71, 3.41, 22.67, 15.65, 1.86, 3.41, 1.89, 1.01, 3.02, 13.81,
    1.55, 1.16, 6.35, 5.6, 2.55, 16.8, 5.48, 1.49, 2.07, 1.05, 1.49, 6.29,
    1.32, 23.22, 1.07, 1.65, 20.07, 1.14, 1.1, 18.38, 4.34, 3.8, 6.17, 2.27,
    1.69, 1.07, 3.74, 1.6, 1.02, 1.45, 1.86, 5.13, 1.57, 6.93, 15.82, 1,
    1.16, 4.14, 1.08, 2.35, 2.15, 13.52, 10.87, 9.85, 1.97, 1, 3.46, 1.31,
    3.28, 2.74, 1.98, 2.22, 1, 9.95, 1.41, 1.43, 2.13, 4.6, 2.68, 4.13,
    1.61, 1.46, 1.23, 9.57, 1.14, 1.17, 14.27, 4.01, 5.55, 1.95, 2.48, 1.78,
    2.21, 1.65, 1.08, 2.63, 8.53, 2.2, 1.33, 21.72, 1.3, 1.43, 6.37, 1.09,
    3.94, 1.88, 3.38, 1.66, 1.41, 22.99, 1.55, 7.5, 25.48, 2.21, 3.62, 1.68,
    9.92, 3.4, 2.66, 1.03, 4.63, 1.89, 1.77, 1.9, 1.01, 1.81, 32.39, 2.1,
    1.23, 6.26, 9.06, 1.17, 2.41, 2.52, 1.63, 5.61, 1, 2.63, 1.88, 1.5,
    23.8, 5.65, 1.05, 1.07, 2.05, 1.7, 2.4, 18.27, 3.68, 13.17, 4.99, 20.81,
    1.51, 6.33, 9.85, 10.15, 17.05, 27.6, 4.65, 3.18, 2.54, 3.92, 4.74,
    1.81, 1.91, 4.42, 1.57, 2.17, 1.25, 1.03, 1.15, 1.19, 13.97, 2.39, 1.34,
    2.52, 1.47, 2.91, 2.31, 1.29, 1.61, 4.13, 1.83, 2.96, 1.08, 1.28, 13.53,
    1.15, 1.51, 1.31, 3.45, 9.32, 5.42, 3.27, 2.56, 2.07, 1.83, 14.1, 15.36,
    1.93, 1.47, 16.96, 1.61, 2.38, 2.66, 1.28, 1.46, 3.09, 6.73, 1.12, 1.85,
    3.21, 1.15, 3.71, 1.64, 4.88, 11.09, 3.82, 2.49, 21.23, 2.01, 2.47,
    2.47, 2.19, 2.14, 1, 2.09, 1.03, 5.22, 1.65, 1.13, 14.43, 1.68, 1.86,
    1.21, 1.14, 1.47, 1.26, 3.44, 23.9, 2.53, 2.72, 1, 1.13, 3.34, 1.43, 1,
    2.48, 2.01, 2.22, 6.43, 1.81, 2.12, 1.3, 4.02, 1.79, 3.9, 1.3, 5.04,
    1.77, 6.67, 2.21, 1.58, 5.38, 2.79, 6.12, 2.95, 1.14, 1.19, 1.19, 10.23,
    17.96, 10.1, 2.4, 9.29, 1.28, 4.07, 1.64, 2.1, 2.67, 1.08, 16.82, 2.83,
    24.42, 1.01, 3.24, 5.05, 3.24, 1.56, 2.32, 1.23, 1.72, 3.39, 1.96, 1.18,
    3.21, 23.95, 9.46, 23.12, 1.45, 3.22, 5, 2.04, 2.73, 6.28, 1.21, 14.3,
    1.48, 3.3, 3.73, 4.09, 2.88, 8.83, 1.15, 4.58, 4.23, 2.34, 2, 11.38,
    1.81, 1.03, 1.76, 2.41, 2.5, 5.82, 2.18, 10.19, 2.08, 18.19, 4.22, 7.78,
    1.96, 1.43, 1.08, 2.38, 1.37, 1.21, 4.48, 1.64, 1.62, 21.24, 1.22, 7.99,
    1.13, 1.29, 2.36, 3.94, 1.08, 1.41, 1.97, 1.41, 1.95, 1.28, 4.56, 3.35,
    1.37, 1.18, 1.03, 3.67, 1.43, 1.8, 2.48, 11.95, 1.5, 3.52, 2.03, 1,
    1.1, 10.13, 1.44, 14.19, 2.1, 8.46, 1.06, 1.66, 1.2, 7.22, 1.75, 1.78,
    3.76, 2.21, 1, 25.19, 5.96, 5.42, 2.67, 1.37, 1.39, 15.95, 2.8, 1.76,
    1.7, 2.81, 8.87, 1.48, 1.03, 1.14, 1.05, 10.29, 1.71, 23.98, 2.34, 1.97,
    1.33, 24.02, 2.01, 13.74, 2.5, 1.33, 1.02, 1.76, 1.37, 8.97, 1.27, 1.38,
    4.47, 1.38, 3.02, 17, 13.35, 1.07, 1.38, 5.74, 6.68, 24.72, 1.47, 1.25,
    4.51, 4.47, 1.99, 1.15, 4.03, 1.17, 3.42, 6.46, 1.31, 1.46, 6.67, 3.79,
    1.56, 3.98, 1.62, 2.13, 1.07, 4.88, 1.62, 1.5, 6.11, 1.31, 1.85, 1.93,
    1.09, 1.49, 1.41, 1.24, 1.05, 6.99, 1.33, 1.73, 10.76, 21.77, 1.18,
    1.06, 5.36, 1.45, 1.16, 6.43, 2.1, 4.15, 1.14, 2.21, 33.48, 2.88, 1,
    4.7, 1.27, 5.75, 4.97, 1.11, 3.51, 21.47, 1.21, 1.98, 1.11, 1.46, 1.77,
    1.22, 2.65, 1.66, 5.29, 1.58, 2.03, 5.86, 1.1, 1.68, 1.35, 1.72, 1.15,
    2.69, 2.81, 3.46, 1.58, 1.07, 7.18, 2.35, 6.05, 1.24, 5.69, 5.46, 1,
    3.04, 4.76, 1.56, 1.41, 2.43, 7.97, 1.22, 1.94, 1.51, 21.71, 3.03, 1.43,
    5.07, 1.87, 1.12, 1, 1.32, 1, 1.08, 1.1, 1.04, 1, 1.09, 1.97, 2.97,
    1.21, 1.61, 5.94, 2.55, 4.48, 1.14, 2.73, 1.34, 1.33, 1.29, 1.25, 5.44,
    1.77, 2.18, 2.52, 1.28, 22.25, 1.04, 3.57, 6.53, 1.34, 5.75, 1.61, 3.89,
    1.07, 2.13, 5.05, 1.53, 3.53, 8.31, 2.15, 1.39, 1.23, 1.68, 17.14, 1.23,
    2.38, 1, 2.02, 19.48, 1.22, 1.42, 6.26, 16.11, 2.05, 3.51, 3.53, 1.83,
    6.86, 1.24, 27.78, 2.33, 3.43, 2.92, 1.26, 15.11, 24.58, 1.12, 2.46,
    5.61, 9.79, 2.33, 1.34, 7.86, 1.1, 2.61, 2.34, 4.5, 1.79, 1.75, 18,
    8.66, 1.92, 11.5, 1.35, 2.53, 1.79, 1.14, 1.58, 1.84, 1.35, 6.44, 4.49,
    3.02, 3.16, 1.12, 1.42, 9.14, 1.26, 1.19, 2.47, 1.2, 3.88, 1.03, 1.85,
    1.07, 1.03, 1.13, 4.87, 1.03, 1.8, 1.29, 6.11, 1.73, 30.16, 2.99, 2.34,
    1.56, 4.33, 1.23, 7.39, 1.57, 3.16, 2.73, 1.46, 1.01, 8.24, 1.61, 2.28,
    1.91, 1.49, 5.12, 3.53, 20.05, 3.26, 2.25, 6.61, 1.35, 4.32, 1, 2.13,
    1.83, 1.26, 2.27, 1.21, 1.64, 1.77, 1.06, 1.05, 1.98, 3.1, 3.74, 22.09,
    2.17, 2.97, 1.26, 1.83, 4.44, 1.08, 2.22, 1.24, 1.7, 20.14, 16.56, 1.72,
    1.37, 1.06, 1.65, 2.42, 3.84, 1, 1.56, 1.93, 1.03, 1.47, 1.76, 12.64,
    1.12, 1.32, 1.89, 1.64, 1.2, 3.15, 1.88, 1.12, 1.01, 1.45, 1.71, 1.65,
    1.65, 5.16, 1.48, 1.73
]

# ============================================================
# دوال التحليل
# ============================================================

@st.cache_data
def compute_basic_stats(data):
    arr = np.array(data)
    return {
        'n'       : len(arr),
        'mean'    : float(np.mean(arr)),
        'median'  : float(np.median(arr)),
        'std'     : float(np.std(arr)),
        'min'     : float(np.min(arr)),
        'max'     : float(np.max(arr)),
        'q25'     : float(np.percentile(arr, 25)),
        'q75'     : float(np.percentile(arr, 75)),
        'skew'    : float(stats.skew(arr)),
        'kurt'    : float(stats.kurtosis(arr)),
    }


@st.cache_data
def run_randomness_tests(data):
    arr = np.array(data)
    results = {}

    # ── 1. اختبار الارتباط الذاتي (Lag-1) ──────────────────
    ac1 = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
    results['autocorr'] = {
        'value'  : round(ac1, 4),
        'pass'   : abs(ac1) < 0.10,
        'label'  : 'ارتباط ذاتي Lag-1',
        'interp' : (
            f"ارتباط ضعيف جداً ({ac1:.4f}) → لا نمط واضح بين قيمة وما يليها"
            if abs(ac1) < 0.10 else
            f"ارتباط ملحوظ ({ac1:.4f}) → يوجد تبعية بين القيم المتتالية"
        )
    }

    # ── 2. Runs Test ─────────────────────────────────────────
    med  = float(np.median(arr))
    runs_seq = [1 if x > med else 0 for x in arr]
    n1 = int(sum(runs_seq))
    n2 = len(runs_seq) - n1
    r  = int(sum(
        1 for i in range(1, len(runs_seq))
        if runs_seq[i] != runs_seq[i-1]
    ) + 1)
    exp_r = 2*n1*n2/(n1+n2) + 1 if (n1+n2) > 0 else 1
    var_r = (
        2*n1*n2*(2*n1*n2 - n1 - n2) /
        ((n1+n2)**2 * (n1+n2-1) + 1e-10)
    )
    z_runs = (r - exp_r) / (var_r**0.5 + 1e-10)
    p_runs = float(2 * (1 - stats.norm.cdf(abs(z_runs))))
    results['runs'] = {
        'z'      : round(z_runs, 4),
        'p'      : round(p_runs, 4),
        'pass'   : p_runs > 0.05,
        'label'  : 'Runs Test (تسلسل)',
        'interp' : (
            f"p={p_runs:.4f} > 0.05 → التسلسل عشوائي"
            if p_runs > 0.05 else
            f"p={p_runs:.4f} < 0.05 → التسلسل غير عشوائي"
        )
    }

    # ── 3. Kolmogorov-Smirnov مقابل توزيع أسي ───────────────
    loc_exp = float(np.min(arr))
    scale_exp = float(np.mean(arr) - loc_exp)
    ks_stat, ks_p = stats.kstest(
        arr, 'expon',
        args=(loc_exp, scale_exp + 1e-10)
    )
    results['ks_exp'] = {
        'stat'   : round(float(ks_stat), 4),
        'p'      : round(float(ks_p),    4),
        'pass'   : ks_p > 0.05,
        'label'  : 'KS - توزيع أسي',
        'interp' : (
            f"p={ks_p:.4f} → البيانات تتبع توزيعاً أسياً"
            if ks_p > 0.05 else
            f"p={ks_p:.4f} → البيانات لا تتبع التوزيع الأسي بدقة"
        )
    }

    # ── 4. اختبار الطبيعي ────────────────────────────────────
    _, norm_p = stats.normaltest(arr)
    results['normality'] = {
        'p'      : round(float(norm_p), 6),
        'pass'   : norm_p > 0.05,
        'label'  : 'اختبار الطبيعية',
        'interp' : (
            "البيانات طبيعية التوزيع"
            if norm_p > 0.05 else
            "البيانات ليست طبيعية (انحراف يميني واضح)"
        )
    }

    # ── 5. ارتباطات متعددة Lag 1-5 ──────────────────────────
    lags = {}
    for lag in range(1, 6):
        if len(arr) > lag:
            c = float(np.corrcoef(arr[:-lag], arr[lag:])[0, 1])
            lags[lag] = round(c, 4)
    results['lags'] = lags

    # ── 6. حكم شامل ─────────────────────────────────────────
    passed = sum([
        results['autocorr']['pass'],
        results['runs']['pass'],
    ])
    results['verdict'] = {
        'random'   : passed >= 2,
        'score'    : passed,
        'max_score': 2
    }

    return results


@st.cache_data
def compute_distribution(data):
    """توزيع البيانات على حالات"""
    arr   = np.array(data)
    total = len(arr)
    bins  = [
        (1.0,  2.0,  'منخفض جداً',  '#4a9eff'),
        (2.0,  5.0,  'منخفض',       '#00d4ff'),
        (5.0,  10.0, 'متوسط',       '#ffa500'),
        (10.0, 20.0, 'مرتفع',       '#ff6b6b'),
        (20.0, 99.0, 'مرتفع جداً',  '#ff0066'),
    ]
    result = []
    for lo, hi, label, color in bins:
        cnt = int(np.sum((arr >= lo) & (arr < hi)))
        result.append({
            'label': label,
            'range': f"{lo}-{hi}",
            'count': cnt,
            'pct'  : round(100 * cnt / total, 2),
            'color': color,
        })
    return result


@st.cache_data
def kelly_criterion(data, multiplier_threshold=2.0):
    """
    حساب Kelly Criterion لإدارة رأس المال
    p  = احتمال الفوز (قيمة >= multiplier_threshold)
    b  = متوسط المضاعف عند الفوز
    q  = 1 - p
    f* = (bp - q) / b
    """
    arr  = np.array(data)
    wins = arr[arr >= multiplier_threshold]
    p    = len(wins) / len(arr)
    q    = 1 - p
    b    = float(np.mean(wins)) if len(wins) > 0 else multiplier_threshold

    kelly = (b * p - q) / (b + 1e-10)
    kelly = max(0.0, min(kelly, 0.5))   # حد أقصى 50% دائماً

    # Half Kelly (أكثر أماناً)
    half_kelly = kelly / 2

    # احتمال الخراب عند استراتيجيات مختلفة
    def ruin_prob(f, p, b, n=100):
        """محاكاة بسيطة لاحتمال الخسارة الكاملة"""
        trials   = 500
        bankrupt = 0
        for _ in range(trials):
            capital = 1.0
            for _ in range(n):
                bet = capital * f
                if np.random.random() < p:
                    capital += bet * b
                else:
                    capital -= bet
                if capital <= 0.01:
                    bankrupt += 1
                    break
        return round(bankrupt / trials * 100, 1)

    return {
        'p'          : round(p,           4),
        'q'          : round(q,           4),
        'b'          : round(b,           4),
        'kelly'      : round(kelly,       4),
        'half_kelly' : round(half_kelly,  4),
        'kelly_pct'  : round(kelly*100,   2),
        'half_pct'   : round(half_kelly*100, 2),
        'threshold'  : multiplier_threshold,
        'n_wins'     : len(wins),
        'n_total'    : len(arr),
    }


@st.cache_data
def simulate_strategies(data, capital=1000.0, n_sim=200):
    """
    محاكاة 3 استراتيجيات رهان على البيانات التاريخية
    1. ثابت (1% من رأس المال)
    2. Kelly نصف
    3. متهور (10%)
    """
    arr = np.array(data)
    threshold = 2.0

    def run_sim(fraction):
        caps = [capital]
        cap  = capital
        for val in arr:
            bet = max(0.01, cap * fraction)
            if val >= threshold:
                cap += bet * (val - 1)
            else:
                cap -= bet
            cap = max(0, cap)
            caps.append(round(cap, 2))
        return caps

    kelly_info = kelly_criterion(data, threshold)
    f_kelly    = kelly_info['half_kelly']

    return {
        'conservative': run_sim(0.01),
        'kelly'       : run_sim(f_kelly),
        'aggressive'  : run_sim(0.10),
        'f_kelly'     : f_kelly,
    }


def stop_loss_analysis(data, capital=1000.0,
                        stop_loss_pct=20.0, take_profit_pct=50.0):
    """تحليل حدود الخسارة والربح"""
    arr = np.array(data)
    sl  = capital * (1 - stop_loss_pct  / 100)
    tp  = capital * (1 + take_profit_pct / 100)

    # محاكاة بسيطة بفرصة ثابتة 1%
    cap  = capital
    hits_sl = 0
    hits_tp = 0
    neutral = 0

    for val in arr:
        bet = cap * 0.02
        if val >= 2.0:
            cap += bet * (val - 1)
        else:
            cap -= bet
        cap = max(0, cap)

        if cap <= sl:
            hits_sl += 1
            cap = capital   # إعادة تعيين
        elif cap >= tp:
            hits_tp += 1
            cap = capital

    total_triggers = hits_sl + hits_tp + neutral
    return {
        'sl_level'   : round(sl,       2),
        'tp_level'   : round(tp,       2),
        'hits_sl'    : hits_sl,
        'hits_tp'    : hits_tp,
        'sl_pct'     : stop_loss_pct,
        'tp_pct'     : take_profit_pct,
    }


# ============================================================
# الشريط الجانبي
# ============================================================
with st.sidebar:
    st.markdown(
        "<h2 style='color:#00d4ff;'>⚙️ الإعدادات</h2>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # مصدر البيانات
    src = st.radio(
        "مصدر البيانات",
        ["البيانات الافتراضية", "إدخال يدوي"],
        index=0
    )

    if src == "إدخال يدوي":
        raw_txt = st.text_area(
            "أدخل القيم (مفصولة بفاصلة أو سطر)",
            height=180,
            placeholder="1.5, 2.3, 8.7 ..."
        )
        import re
        nums = re.findall(r"[\d.]+", raw_txt)
        try:
            user_data = [float(x) for x in nums if float(x) > 0]
            if len(user_data) < 30:
                st.warning("يُنصح بـ 30 قيمة على الأقل")
                user_data = DEFAULT_DATA
        except Exception:
            user_data = DEFAULT_DATA
    else:
        user_data = DEFAULT_DATA

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#87ceeb;'>💰 إعدادات رأس المال</h3>",
        unsafe_allow_html=True
    )

    capital      = st.number_input(
        "رأس المال الابتدائي", 100, 1_000_000, 1000, 100
    )
    mult_thresh  = st.slider(
        "عتبة الفوز (المضاعف)", 1.1, 5.0, 2.0, 0.1,
        help="القيمة الدنيا التي تُعتبر 'فوزاً'"
    )
    sl_pct       = st.slider(
        "حد الخسارة Stop Loss %", 5, 50, 20, 5
    )
    tp_pct       = st.slider(
        "هدف الربح Take Profit %", 10, 200, 50, 10
    )

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#87ceeb;'>📊 إعدادات العرض</h3>",
        unsafe_allow_html=True
    )
    n_show = st.slider("آخر N قيمة للعرض", 50, 500, 150, 50)

    st.markdown("---")
    st.markdown("""
    <div style='color:#4a7a9b; font-size:0.8em; line-height:1.6;'>
        ⚠️ <strong style='color:#ffa500;'>تحذير قانوني:</strong><br>
        هذه الأداة للتحليل الإحصائي فقط.<br>
        لا تضمن أرباحاً ولا تتنبأ بدقة مطلقة.<br>
        المقامرة تنطوي على مخاطر مالية حقيقية.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# تشغيل الحسابات
# ============================================================
data         = user_data
basic        = compute_basic_stats(data)
rand_tests   = run_randomness_tests(data)
dist_info    = compute_distribution(data)
kelly_info   = kelly_criterion(data, mult_thresh)
sim_data     = simulate_strategies(data, capital)
sl_info      = stop_loss_analysis(data, capital, sl_pct, tp_pct)

# ============================================================
# الرأس
# ============================================================
st.markdown("""
<h1 style='text-align:center; color:#00d4ff; margin-bottom:4px;'>
    📊 المحلل الإحصائي الصادق
</h1>
<p style='text-align:center; color:#87ceeb; font-size:1.05em;'>
    تحليل عشوائية البيانات · إدارة رأس المال · حدود الخسارة والربح
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# بطاقات الإحصاءات
# ============================================================
c1, c2, c3, c4, c5, c6 = st.columns(6)
cards = [
    (c1, len(data),             "عدد القيم",          ""),
    (c2, f"{basic['mean']:.2f}","المتوسط",            ""),
    (c3, f"{basic['median']:.2f}","الوسيط",           ""),
    (c4, f"{basic['std']:.2f}", "الانحراف المعياري",  ""),
    (c5, f"{basic['max']:.2f}", "أعلى قيمة",          ""),
    (c6, f"{basic['skew']:.2f}","معامل الانحراف",     ""),
]
for col, val, lbl, sub in cards:
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">{lbl}</div>
            <div class="card-value">{val}</div>
            <div class="card-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# التبويبات الرئيسية
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 اختبار العشوائية",
    "📈 تحليل البيانات",
    "💰 Kelly & رأس المال",
    "🛡️ Stop Loss / Take Profit",
    "📋 التوصيات الشاملة",
])

# ══════════════════════════════════════════════════════════════
# التبويب 1: اختبار العشوائية
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "<div class='section-title'>🔬 نتائج اختبارات العشوائية</div>",
        unsafe_allow_html=True
    )

    # الحكم الشامل
    verdict   = rand_tests['verdict']
    is_random = verdict['random']
    v_color   = "#ffa500" if is_random else "#00ff88"
    v_icon    = "⚠️" if is_random else "🔍"
    v_text    = (
        "البيانات تبدو عشوائية - لا نمط حتمي قابل للاستغلال"
        if is_random else
        "تم اكتشاف انحراف عن العشوائية - يستحق دراسة أعمق"
    )

    st.markdown(f"""
    <div style="background:rgba({
        '255,165,0' if is_random else '0,255,136'
    },0.08); border:2px solid {v_color};
    border-radius:14px; padding:20px; text-align:center; margin-bottom:20px;">
        <div style="font-size:2em;">{v_icon}</div>
        <div style="color:{v_color}; font-size:1.3em;
                    font-weight:bold; margin:8px 0;">
            {v_text}
        </div>
        <div style="color:#87ceeb; font-size:0.9em;">
            اجتاز {verdict['score']} من {verdict['max_score']} اختبارات العشوائية
        </div>
    </div>""", unsafe_allow_html=True)

    # نتائج الاختبارات التفصيلية
    tests_to_show = ['autocorr', 'runs', 'ks_exp', 'normality']
    test_cols     = st.columns(2)

    for i, key in enumerate(tests_to_show):
        t   = rand_tests[key]
        col = test_cols[i % 2]
        with col:
            passed     = t.get('pass', False)
            box_class  = "result-pass" if passed else "result-fail"
            icon       = "✅" if passed else "❌"

            # القيمة الرئيسية للعرض
            if 'value' in t:
                main_val = f"{t['value']}"
            elif 'z' in t:
                main_val = f"z = {t['z']}, p = {t['p']}"
            elif 'stat' in t:
                main_val = f"stat = {t['stat']}, p = {t['p']}"
            else:
                main_val = f"p = {t['p']}"

            st.markdown(f"""
            <div class="{box_class}">
                <strong>{icon} {t['label']}</strong><br>
                <span style="font-size:0.9em; opacity:0.9;">
                    {main_val}
                </span><br>
                <span style="font-size:0.85em; opacity:0.8;
                             margin-top:4px; display:block;">
                    {t['interp']}
                </span>
            </div>""", unsafe_allow_html=True)

    # رسم الارتباطات الذاتية
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>📉 الارتباط الذاتي (Lag 1-5)</div>",
        unsafe_allow_html=True
    )

    lags   = rand_tests['lags']
    lag_x  = list(lags.keys())
    lag_y  = list(lags.values())
    colors = ['#00ff88' if abs(v) < 0.10 else '#ff4444' for v in lag_y]

    fig_lag = go.Figure()
    fig_lag.add_trace(go.Bar(
        x=[f"Lag {k}" for k in lag_x],
        y=lag_y,
        marker_color=colors,
        text=[f"{v:.4f}" for v in lag_y],
        textposition='outside'
    ))
    fig_lag.add_hline(y=0.10, line_dash='dash',
                      line_color='#ffa500', opacity=0.7,
                      annotation_text="حد الأهمية +0.10")
    fig_lag.add_hline(y=-0.10, line_dash='dash',
                      line_color='#ffa500', opacity=0.7,
                      annotation_text="حد الأهمية -0.10")
    fig_lag.update_layout(
        title="معاملات الارتباط الذاتي - كلما اقتربت من 0 كلما كانت العشوائية أعلى",
        yaxis_title="معامل الارتباط",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        height=320,
    )
    st.plotly_chart(fig_lag, use_container_width=True)

    # توزيع القيم - هيستوغرام
    st.markdown(
        "<div class='section-title'>📊 توزيع القيم</div>",
        unsafe_allow_html=True
    )
    arr_np  = np.array(data)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=arr_np,
        nbinsx=40,
        marker_color='#4a9eff',
        opacity=0.8,
        name='التوزيع الفعلي'
    ))
    # منحنى أسي للمقارنة
    x_range = np.linspace(float(arr_np.min()),
                          float(arr_np.max()), 200)
    loc_e   = float(arr_np.min())
    scl_e   = float(arr_np.mean()) - loc_e
    pdf_e   = stats.expon.pdf(x_range, loc=loc_e, scale=scl_e+1e-10)
    pdf_e   = pdf_e * len(data) * (arr_np.max()-arr_np.min()) / 40

    fig_hist.add_trace(go.Scatter(
        x=x_range, y=pdf_e,
        mode='lines', name='توزيع أسي مرجعي',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    fig_hist.update_layout(
        title="توزيع القيم مقارنةً بالتوزيع الأسي المرجعي",
        xaxis_title="القيمة",
        yaxis_title="التكرار",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0.3)'),
        height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# التبويب 2: تحليل البيانات
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<div class='section-title'>📈 السلسلة الزمنية</div>",
        unsafe_allow_html=True
    )

    data_slice = data[-n_show:]
    idx_slice  = list(range(len(data) - n_show, len(data)))

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=idx_slice, y=data_slice,
        mode='lines',
        line=dict(color='#4a9eff', width=1),
        name='القيم', opacity=0.8
    ))

    # تلوين المناطق
    for lo, hi, lbl, clr in [
        (1, 2,  'منخفض جداً', 'rgba(74,158,255,0.08)'),
        (2, 5,  'منخفض',      'rgba(0,212,255,0.05)'),
        (5, 10, 'متوسط',      'rgba(255,165,0,0.05)'),
    ]:
        fig_ts.add_hrect(
            y0=lo, y1=hi,
            fillcolor=clr,
            line_width=0,
            annotation_text=lbl,
            annotation_position="left"
        )

    # تمييز القيم العالية
    hi_idx = [idx_slice[i] for i, v in enumerate(data_slice) if v >= 10]
    hi_val = [v for v in data_slice if v >= 10]
    fig_ts.add_trace(go.Scatter(
        x=hi_idx, y=hi_val,
        mode='markers',
        marker=dict(color='#ff6b6b', size=7),
        name='قيم عالية (≥10)'
    ))

    fig_ts.add_hline(
        y=basic['mean'], line_dash='dot',
        line_color='#ffa500', opacity=0.7,
        annotation_text=f"المتوسط {basic['mean']:.2f}"
    )
    fig_ts.update_layout(
        title=f"آخر {n_show} قيمة",
        xaxis_title="الموضع",
        yaxis_title="القيمة",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0.3)'),
        height=400,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # توزيع الحالات
    st.markdown(
        "<div class='section-title'>🎨 توزيع الحالات</div>",
        unsafe_allow_html=True
    )
    d_col1, d_col2 = st.columns([1, 1])

    with d_col1:
        labels = [d['label'] for d in dist_info]
        values = [d['count'] for d in dist_info]
        colors = [d['color'] for d in dist_info]

        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.45,
            textinfo='label+percent',
            textfont=dict(size=12, color='white'),
        ))
        fig_pie.update_layout(
            title="توزيع القيم على الحالات",
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            legend=dict(bgcolor='rgba(0,0,0,0.3)'),
            height=360,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with d_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        for d in dist_info:
            bar_w = int(d['pct'] * 2)
            st.markdown(f"""
            <div style="margin:10px 0;">
                <div style="display:flex; justify-content:space-between;
                            margin-bottom:4px;">
                    <span style="color:{d['color']};">
                        {d['label']} ({d['range']}x)
                    </span>
                    <span style="color:#00d4ff; font-weight:bold;">
                        {d['count']} ({d['pct']}%)
                    </span>
                </div>
                <div style="background:#1a3a5c; border-radius:4px; height:10px;">
                    <div style="background:{d['color']}; width:{min(bar_w,200)}px;
                                height:10px; border-radius:4px;
                                max-width:100%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # المتوسط المتحرك
    st.markdown(
        "<div class='section-title'>📉 المتوسط المتحرك</div>",
        unsafe_allow_html=True
    )
    win = st.slider("نافذة المتوسط المتحرك", 5, 50, 20, 5)
    arr_np = np.array(data)
    ma     = np.convolve(arr_np, np.ones(win)/win, mode='valid')
    ma_idx = list(range(win - 1, len(data)))

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=list(range(len(data))), y=list(data),
        mode='lines', opacity=0.3,
        line=dict(color='#4a9eff', width=1),
        name='القيم الأصلية'
    ))
    fig_ma.add_trace(go.Scatter(
        x=ma_idx, y=list(ma),
        mode='lines',
        line=dict(color='#00ff88', width=2),
        name=f'MA({win})'
    ))
    fig_ma.add_hline(
        y=float(np.mean(arr_np)), line_dash='dash',
        line_color='#ffa500', opacity=0.6,
        annotation_text="المتوسط الكلي"
    )
    fig_ma.update_layout(
        title=f"المتوسط المتحرك (نافذة={win})",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0.3)'),
        height=320,
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    # جدول الإحصاءات
    with st.expander("📋 جدول الإحصاءات التفصيلية"):
        stats_df = pd.DataFrame([
            {"المقياس": "العدد",               "القيمة": len(data)},
            {"المقياس": "المتوسط",             "القيمة": round(basic['mean'],   4)},
            {"المقياس": "الوسيط",              "القيمة": round(basic['median'], 4)},
            {"المقياس": "الانحراف المعياري",   "القيمة": round(basic['std'],    4)},
            {"المقياس": "الحد الأدنى",         "القيمة": round(basic['min'],    4)},
            {"المقياس": "الربيع الأول (Q1)",   "القيمة": round(basic['q25'],   4)},
            {"المقياس": "الربيع الثالث (Q3)",  "القيمة": round(basic['q75'],   4)},
            {"المقياس": "الحد الأقصى",         "القيمة": round(basic['max'],    4)},
            {"المقياس": "معامل الانحراف",      "القيمة": round(basic['skew'],   4)},
            {"المقياس": "التفرطح (Kurtosis)",  "القيمة": round(basic['kurt'],   4)},
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# التبويب 3: Kelly & رأس المال
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "<div class='section-title'>📐 حساب Kelly Criterion</div>",
        unsafe_allow_html=True
    )

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(f"""
        <div class="kelly-box">
            <div style="color:#87ceeb; font-size:0.85em;">Kelly الكامل</div>
            <div class="kelly-value">{kelly_info['kelly_pct']:.1f}%</div>
            <div style="color:#4a7a9b; font-size:0.8em; margin-top:6px;">
                من رأس المال لكل رهان
            </div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kelly-box" style="border-color:#00d4ff;">
            <div style="color:#87ceeb; font-size:0.85em;">
                نصف Kelly (الموصى به)
            </div>
            <div class="kelly-value" style="color:#00d4ff;">
                {kelly_info['half_pct']:.1f}%
            </div>
            <div style="color:#4a7a9b; font-size:0.8em; margin-top:6px;">
                أكثر أماناً وعملية
            </div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kelly-box" style="border-color:#ffa500;">
            <div style="color:#87ceeb; font-size:0.85em;">
                احتمال الفوز (≥{mult_thresh}x)
            </div>
            <div class="kelly-value" style="color:#ffa500;">
                {kelly_info['p']*100:.1f}%
            </div>
            <div style="color:#4a7a9b; font-size:0.8em; margin-top:6px;">
                {kelly_info['n_wins']} من {kelly_info['n_total']} قيمة
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # شرح Kelly
    with st.expander("📖 كيف يعمل Kelly Criterion؟"):
        st.markdown(f"""
        **معادلة Kelly:**
        ```
        f* = (b × p - q) / b
        ```
        حيث:
        - **p** = احتمال الفوز = **{kelly_info['p']:.4f}**
        - **q** = احتمال الخسارة = **{kelly_info['q']:.4f}**
        - **b** = متوسط المضاعف عند الفوز = **{kelly_info['b']:.4f}**
        - **f*** = نسبة رأس المال المُراهن = **{kelly_info['kelly_pct']:.2f}%**

        **لماذا نصف Kelly؟**
        - Kelly الكامل يعطي أعلى نمو نظرياً
        - لكنه ينطوي على تذبذب عالٍ جداً
        - نصف Kelly يقلل التذبذب بـ 50% مع تقليل النمو بـ 25% فقط
        """)

    # محاكاة الاستراتيجيات
    st.markdown(
        "<div class='section-title'>📊 مقارنة الاستراتيجيات</div>",
        unsafe_allow_html=True
    )

    sim   = sim_data
    x_sim = list(range(len(sim['conservative'])))

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=x_sim, y=sim['conservative'],
        mode='lines', name='محافظ (1%)',
        line=dict(color='#4a9eff', width=1.5)
    ))
    fig_sim.add_trace(go.Scatter(
        x=x_sim, y=sim['kelly'],
        mode='lines',
        name=f'نصف Kelly ({sim["f_kelly"]*100:.1f}%)',
        line=dict(color='#00ff88', width=2)
    ))
    fig_sim.add_trace(go.Scatter(
        x=x_sim, y=sim['aggressive'],
        mode='lines', name='متهور (10%)',
        line=dict(color='#ff4444', width=1.5)
    ))
    fig_sim.add_hline(
        y=capital, line_dash='dot',
        line_color='white', opacity=0.3,
        annotation_text="رأس المال الأصلي"
    )
    fig_sim.update_layout(
        title="محاكاة رأس المال عبر الزمن - 3 استراتيجيات",
        xaxis_title="الجولة",
        yaxis_title="رأس المال",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0.3)'),
        height=420,
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    # نتائج المحاكاة
    sim_results = st.columns(3)
    strategies  = [
        ("محافظ 1%",
         sim['conservative'],
         "#4a9eff"),
        (f"نصف Kelly {sim['f_kelly']*100:.1f}%",
         sim['kelly'],
         "#00ff88"),
        ("متهور 10%",
         sim['aggressive'],
         "#ff4444"),
    ]
    for col, (name, caps, clr) in zip(sim_results, strategies):
        final     = caps[-1]
        change    = (final - capital) / capital * 100
        min_cap   = min(caps)
        drawdown  = (min_cap - capital) / capital * 100
        with col:
            st.markdown(f"""
            <div class="card" style="border-color:{clr}40;">
                <div class="card-title">{name}</div>
                <div class="card-value" style="color:{clr};">
                    {final:,.0f}
                </div>
                <div class="card-sub">
                    التغير: {change:+.1f}%<br>
                    أدنى نقطة: {min_cap:,.0f}
                    ({drawdown:.1f}%)
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# التبويب 4: Stop Loss / Take Profit
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        "<div class='section-title'>"
        "🛡️ تحليل حدود الخسارة والربح"
        "</div>",
        unsafe_allow_html=True
    )

    sl_col1, sl_col2 = st.columns(2)

    with sl_col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔴 Stop Loss</div>
            <div class="card-value" style="color:#ff4444;">
                {sl_info['sl_level']:,.0f}
            </div>
            <div class="card-sub">
                -{sl_pct}% من رأس المال ({capital:,.0f})<br>
                تفعيل عند الوصول لهذا المستوى
            </div>
        </div>""", unsafe_allow_html=True)

    with sl_col2:
        st.markdown(f"""
        <div class="card" style="border-color:#00ff88;">
            <div class="card-title">🟢 Take Profit</div>
            <div class="card-value" style="color:#00ff88;">
                {sl_info['tp_level']:,.0f}
            </div>
            <div class="card-sub">
                +{tp_pct}% من رأس المال ({capital:,.0f})<br>
                جني الأرباح عند هذا المستوى
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # رسم مناطق الخطر والأمان
    sl_val = sl_info['sl_level']
    tp_val = sl_info['tp_level']

    fig_zones = go.Figure()

    # منطقة الخطر
    fig_zones.add_hrect(
        y0=0, y1=sl_val,
        fillcolor='rgba(255,68,68,0.08)',
        line_width=0,
        annotation_text="🔴 منطقة الخطر",
        annotation_position="left"
    )
    # منطقة الأمان
    fig_zones.add_hrect(
        y0=sl_val, y1=tp_val,
        fillcolor='rgba(74,158,255,0.05)',
        line_width=0,
        annotation_text="🔵 منطقة الأمان",
        annotation_position="left"
    )
    # منطقة الربح
    fig_zones.add_hrect(
        y0=tp_val, y1=tp_val * 2,
        fillcolor='rgba(0,255,136,0.05)',
        line_width=0,
        annotation_text="🟢 منطقة الربح",
        annotation_position="left"
    )

    # محاكاة رأس المال
    cap_trace = []
    cap_run   = float(capital)
    for val in data[-200:]:
        bet = cap_run * kelly_info['half_kelly']
        if val >= mult_thresh:
            cap_run += bet * (val - 1)
        else:
            cap_run -= bet
        cap_run = max(0, cap_run)
        cap_trace.append(round(cap_run, 2))

    fig_zones.add_trace(go.Scatter(
        x=list(range(len(cap_trace))),
        y=cap_trace,
        mode='lines',
        name='رأس المال',
        line=dict(color='#00d4ff', width=2)
    ))
    fig_zones.add_hline(
        y=sl_val, line_dash='dash',
        line_color='#ff4444', line_width=2,
        annotation_text=f"Stop Loss: {sl_val:,.0f}"
    )
    fig_zones.add_hline(
        y=float(capital), line_dash='dot',
        line_color='white', opacity=0.4,
        annotation_text=f"البداية: {capital:,.0f}"
    )
    fig_zones.add_hline(
        y=tp_val, line_dash='dash',
        line_color='#00ff88', line_width=2,
        annotation_text=f"Take Profit: {tp_val:,.0f}"
    )
    fig_zones.update_layout(
        title="محاكاة رأس المال مع مناطق Stop Loss / Take Profit",
        xaxis_title="الجولة",
        yaxis_title="رأس المال",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,25,45,0.8)',
        font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0.3)'),
        height=420,
    )
    st.plotly_chart(fig_zones, use_container_width=True)

    # إرشادات Stop Loss
    st.markdown(
        "<div class='section-title'>📚 إرشادات إدارة رأس المال</div>",
        unsafe_allow_html=True
    )
    guidelines = [
        ("🔴 Stop Loss إلزامي",
         f"أوقف اللعب فوراً عند خسارة {sl_pct}% "
         f"({sl_val:,.0f}). لا استثناءات.",
         "rec-danger"),
        ("🟢 Take Profit منضبط",
         f"اسحب الأرباح أو توقف عند تحقيق {tp_pct}% "
         f"ربح ({tp_val:,.0f}). لا تنتظر أكثر.",
         "rec-warn"),
        ("💰 حجم الرهان الثابت",
         f"لا تتجاوز {kelly_info['half_pct']:.1f}% من رأس المال "
         f"لكل جولة (نصف Kelly).",
         "rec-strong"),
        ("⏰ حدود زمنية",
         "حدد عدداً أقصى من الجولات قبل البدء والتزم به.",
         "rec-warn"),
        ("🧠 القرار البارد",
         "لا تزد رهانك بعد الخسارة. الخسارة السابقة "
         "لا تغير احتمالات المستقبل.",
         "rec-danger"),
    ]
    for title, body, cls in guidelines:
        st.markdown(f"""
        <div class="{cls}">
            <strong>{title}</strong><br>
            <span style="font-size:0.9em; opacity:0.9;">{body}</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# التبويب 5: التوصيات الشاملة
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        "<div class='section-title'>📋 لوحة التوصيات الشاملة</div>",
        unsafe_allow_html=True
    )

    # ملخص الوضع
    is_rand    = rand_tests['verdict']['random']
    k_pct      = kelly_info['half_pct']
    win_rate   = kelly_info['p'] * 100
    avg_win    = kelly_info['b']

    # مؤشر الجودة الإجمالية
    quality_score = 0
    if win_rate >= 30:
        quality_score += 25
    if avg_win >= 2.5:
        quality_score += 25
    if k_pct >= 2:
        quality_score += 25
    if not is_rand:
        quality_score += 25

    q_color = (
        "#00ff88" if quality_score >= 75 else
        "#ffa500" if quality_score >= 50 else
        "#ff4444"
    )
    q_label = (
        "ظروف جيدة نسبياً" if quality_score >= 75 else
        "ظروف متوسطة"      if quality_score >= 50 else
        "ظروف صعبة"
    )

    # مؤشر Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = quality_score,
        title = {'text': "مؤشر جودة الظروف",
                 'font': {'color': 'white', 'size': 16}},
        number= {'suffix': "/100",
                 'font': {'color': q_color, 'size': 32}},
        gauge = {
            'axis' : {'range': [0, 100], 'tickcolor': 'gray'},
            'bar'  : {'color': q_color},
            'steps': [
                {'range': [0,  50], 'color': 'rgba(255,68,68,0.15)'},
                {'range': [50, 75], 'color': 'rgba(255,165,0,0.15)'},
                {'range': [75,100], 'color': 'rgba(0,255,136,0.15)'},
            ],
            'bgcolor': 'rgba(0,0,0,0)'
        }
    ))
    fig_gauge.update_layout(
        height=260,
        margin=dict(t=60, b=10, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    g_col, s_col = st.columns([1, 1])

    with g_col:
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center; color:{q_color}; "
            f"font-size:1.2em; font-weight:bold;'>{q_label}</div>",
            unsafe_allow_html=True
        )

    with s_col:
        summary_items = [
            ("احتمال الفوز (≥{:.1f}x)".format(mult_thresh),
             f"{win_rate:.1f}%",
             "#ffa500"),
            ("متوسط المضاعف عند الفوز",
             f"{avg_win:.2f}x",
             "#00d4ff"),
            ("نصف Kelly الموصى به",
             f"{k_pct:.2f}%",
             "#00ff88"),
            ("البيانات عشوائية؟",
             "نعم ⚠️" if is_rand else "غير مؤكد 🔍",
             "#ffa500" if is_rand else "#ff6b6b"),
            ("Stop Loss عند",
             f"{sl_info['sl_level']:,.0f} (-{sl_pct}%)",
             "#ff4444"),
            ("Take Profit عند",
             f"{sl_info['tp_level']:,.0f} (+{tp_pct}%)",
             "#00ff88"),
        ]
        st.markdown("<br>", unsafe_allow_html=True)
        for label, value, color in summary_items:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        padding:8px 4px;
                        border-bottom:1px solid #1a3a5c;">
                <span style="color:#87ceeb;">{label}</span>
                <span style="color:{color}; font-weight:bold;">
                    {value}
                </span>
            </div>""", unsafe_allow_html=True)

    # التوصيات التفصيلية
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🎯 التوصيات التفصيلية</div>",
        unsafe_allow_html=True
    )

    recs = []

    # توصية 1: حجم الرهان
    if k_pct < 1:
        recs.append((
            "rec-danger",
            "⛔ لا يُنصح بالمشاركة",
            f"Kelly يعطي {k_pct:.2f}% فقط مما يعني أن الأفضلية "
            "ليست في صالحك. المشاركة بأي مبلغ ليست مُجدية إحصائياً."
        ))
    elif k_pct < 3:
        recs.append((
            "rec-warn",
            "⚠️ رهانات صغيرة جداً فقط",
            f"Kelly = {k_pct:.2f}% → ارهن بـ {k_pct:.1f}% من رأس المال "
            f"كحد أقصى ({capital * k_pct/100:,.0f} لكل جولة)."
        ))
    else:
        recs.append((
            "rec-strong",
            "✅ حجم رهان معقول",
            f"Kelly = {k_pct:.2f}% → ارهن بـ {k_pct:.1f}% "
            f"({capital * k_pct/100:,.0f} لكل جولة). لا تتجاوز هذا."
        ))

    # توصية 2: احتمال الفوز
    if win_rate < 20:
        recs.append((
            "rec-danger",
            "🔴 احتمال فوز منخفض جداً",
            f"فقط {win_rate:.1f}% من القيم تتجاوز {mult_thresh}x. "
            "هذا يعني خسارة في 4 من كل 5 جولات تقريباً."
        ))
    elif win_rate < 35:
        recs.append((
            "rec-warn",
            "🟡 احتمال فوز متوسط",
            f"{win_rate:.1f}% من القيم تتجاوز {mult_thresh}x. "
            "المفتاح هو ضبط حجم الرهان بدقة."
        ))
    else:
        recs.append((
            "rec-strong",
            "🟢 احتمال فوز معقول",
            f"{win_rate:.1f}% من القيم تتجاوز {mult_thresh}x. "
            "لكن تذكر: الماضي لا يضمن المستقبل."
        ))

    # توصية 3: العشوائية
    if is_rand:
        recs.append((
            "rec-warn",
            "⚠️ البيانات تبدو عشوائية",
            "اجتازت البيانات اختبارات العشوائية. هذا يعني أن "
            "أي نمط تراه قد يكون وهماً إحصائياً (Pareidolia). "
            "لا تعتمد على أنماط بصرية."
        ))
    else:
        recs.append((
            "rec-warn",
            "🔍 تم رصد انحراف عن العشوائية",
            "بعض الاختبارات أظهرت انحرافاً. لكن هذا لا يعني "
            "إمكانية التنبؤ المضمون. يحتاج دراسة أعمق بمزيد من البيانات."
        ))

    # توصية 4: إدارة رأس المال
    recs.append((
        "rec-strong",
        "💡 القاعدة الذهبية لإدارة رأس المال",
        f"Stop Loss عند {sl_info['sl_level']:,.0f} (-{sl_pct}%) | "
        f"Take Profit عند {sl_info['tp_level']:,.0f} (+{tp_pct}%) | "
        f"لا تتجاوز {k_pct:.1f}% لكل رهان. هذه الثلاثة معاً هي درعك الواقي."
    ))

    # توصية 5: تحذير نهائي
    recs.append((
        "rec-danger",
        "🚨 تحذير: مغالطة القمار",
        "الخسارة المتتالية لا تعني أن الفوز 'واجب' قادماً. "
        "كل جولة مستقلة إحصائياً. لا تزد رهانك بعد الخسارة أبداً. "
        "هذا هو أكثر الأخطاء تدميراً."
    ))

    for cls, title, body in recs:
        st.markdown(f"""
        <div class="{cls}" style="margin:10px 0;">
            <strong style="font-size:1.05em;">{title}</strong><br>
            <span style="font-size:0.92em; opacity:0.9;
                         line-height:1.5;">{body}</span>
        </div>""", unsafe_allow_html=True)

    # حاسبة رأس المال التفاعلية
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🧮 حاسبة الرهان التفاعلية</div>",
        unsafe_allow_html=True
    )

    calc_col1, calc_col2 = st.columns(2)
    with calc_col1:
        my_capital = st.number_input(
            "رأس مالي الحالي", 10, 1_000_000,
            int(capital), 10
        )
        my_strategy = st.selectbox(
            "الاستراتيجية",
            ["محافظ (0.5%)", "نصف Kelly", "كامل Kelly", "مخصص"]
        )
        if my_strategy == "مخصص":
            custom_pct = st.slider("نسبة مخصصة %", 0.1, 25.0, 2.0, 0.1)
        else:
            custom_pct = None

    with calc_col2:
        if my_strategy == "محافظ (0.5%)":
            bet_pct = 0.5
        elif my_strategy == "نصف Kelly":
            bet_pct = kelly_info['half_pct']
        elif my_strategy == "كامل Kelly":
            bet_pct = kelly_info['kelly_pct']
        else:
            bet_pct = custom_pct or 2.0

        bet_amount = my_capital * bet_pct / 100
        sl_amount  = my_capital * sl_pct  / 100
        tp_amount  = my_capital * tp_pct  / 100

        st.markdown(f"""
        <div class="card" style="margin-top:10px;">
            <div class="card-title">نتائج الحاسبة</div>
            <br>
            <div style="display:flex; justify-content:space-between;
                        padding:6px 0; border-bottom:1px solid #1a3a5c;">
                <span style="color:#87ceeb;">مبلغ الرهان</span>
                <span style="color:#00ff88; font-weight:bold;">
                    {bet_amount:,.2f} ({bet_pct:.2f}%)
                </span>
            </div>
            <div style="display:flex; justify-content:space-between;
                        padding:6px 0; border-bottom:1px solid #1a3a5c;">
                <span style="color:#87ceeb;">حد الخسارة</span>
                <span style="color:#ff4444; font-weight:bold;">
                    -{sl_amount:,.2f}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between;
                        padding:6px 0; border-bottom:1px solid #1a3a5c;">
                <span style="color:#87ceeb;">هدف الربح</span>
                <span style="color:#00ff88; font-weight:bold;">
                    +{tp_amount:,.2f}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between;
                        padding:6px 0;">
                <span style="color:#87ceeb;">عدد الرهانات حتى Stop Loss</span>
                <span style="color:#ffa500; font-weight:bold;">
                    ~{int(sl_amount / (bet_amount + 0.01))} جولة
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

# ============================================================
# الفوتر
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#2a4a6a; font-size:0.82em;
            line-height:1.8;">
    📊 المحلل الإحصائي الصادق | بُني بـ Python & Streamlit<br>
    ⚠️ للأغراض التعليمية والتحليلية فقط ·
    المقامرة تنطوي على مخاطر مالية حقيقية ·
    لا تستثمر ما لا تستطيع خسارته
</div>
""", unsafe_allow_html=True)
