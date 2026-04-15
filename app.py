# -*- coding: utf-8 -*-
# app.py — مشروع تخرج متقدم: تحليل PRNG + كشف أنماط + تنبؤ

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import Counter, defaultdict
from itertools import product
import json
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🎓 PRNG Analysis & Pattern Predictor",
    page_icon="🎓",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
#                   دوال مساعدة
# ══════════════════════════════════════════════════════════════
def to_python(obj):
    """تحويل numpy → Python قياسي لـ JSON"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python(i) for i in obj]
    return obj


def categorize(val: float) -> str:
    """تصنيف القيمة إلى فئة"""
    if val < 1.5:
        return 'VERY_LOW'
    elif val < 2.0:
        return 'LOW'
    elif val < 3.0:
        return 'MED'
    elif val < 5.0:
        return 'HIGH'
    elif val < 10.0:
        return 'VERY_HIGH'
    else:
        return 'EXTREME'


CAT_ORDER  = ['VERY_LOW','LOW','MED','HIGH','VERY_HIGH','EXTREME']
CAT_LABELS = {
    'VERY_LOW' : '< 1.5x',
    'LOW'      : '1.5-2x',
    'MED'      : '2-3x',
    'HIGH'     : '3-5x',
    'VERY_HIGH': '5-10x',
    'EXTREME'  : '> 10x'
}
CAT_COLORS = {
    'VERY_LOW' : '#e74c3c',
    'LOW'      : '#e67e22',
    'MED'      : '#f1c40f',
    'HIGH'     : '#2ecc71',
    'VERY_HIGH': '#3498db',
    'EXTREME'  : '#9b59b6'
}


# ══════════════════════════════════════════════════════════════
#              1. اختبارات NIST للعشوائية
# ══════════════════════════════════════════════════════════════
class RandomnessTestSuite:

    def __init__(self, data: list):
        self.raw      = np.array(data, dtype=float)
        self.binary   = (self.raw >= 2.0).astype(int)
        self.log_data = np.log(np.maximum(self.raw, 1.0))
        self.results  = {}

    def frequency_test(self) -> dict:
        n  = len(self.binary)
        n1 = int(self.binary.sum())
        n0 = int(n - n1)
        s  = float(abs(n1 - n0) / np.sqrt(n))
        p  = float(2 * (1 - stats.norm.cdf(s)))
        r  = {
            'test_name'     : 'Frequency (Monobit) Test',
            'n_high'        : n1,
            'n_low'         : n0,
            'ratio_high'    : round(float(n1 / n), 4),
            'statistic'     : round(s, 4),
            'p_value'       : round(p, 4),
            'passed'        : bool(p >= 0.01),
            'interpretation': (
                '✅ التوزيع متوازن'
                if p >= 0.01 else '🔴 خلل في التوزيع!'
            )
        }
        self.results['frequency'] = r
        return r

    def runs_test(self) -> dict:
        n  = len(self.binary)
        pi = float(self.binary.mean())
        if abs(pi - 0.5) > 2 / np.sqrt(n):
            r = {
                'test_name'     : 'Runs Test',
                'passed'        : False,
                'p_value'       : 0.0,
                'statistic'     : 0.0,
                'runs_observed' : 0,
                'runs_expected' : 0.0,
                'interpretation': '🔴 نسبة غير متوازنة'
            }
            self.results['runs'] = r
            return r
        runs = int(1 + sum(
            1 for i in range(1, n)
            if self.binary[i] != self.binary[i-1]
        ))
        exp = float(2 * n * pi * (1 - pi))
        var = float(
            2 * n * pi * (1-pi) * (2*pi*(1-pi) - 1/n)
        )
        z = float((runs - exp) / np.sqrt(var)) if var > 0 else 0.0
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        r = {
            'test_name'     : 'Runs Test',
            'runs_observed' : runs,
            'runs_expected' : round(exp, 2),
            'statistic'     : round(z, 4),
            'p_value'       : round(p, 4),
            'passed'        : bool(p >= 0.01),
            'interpretation': (
                '✅ التسلسلات طبيعية'
                if p >= 0.01 else '🔴 أنماط في التسلسلات!'
            )
        }
        self.results['runs'] = r
        return r

    def autocorrelation_test(self, max_lag: int = 30) -> dict:
        n     = len(self.binary)
        bound = float(1.96 / np.sqrt(n))
        acfs  = []
        sig   = []
        for lag in range(1, min(max_lag+1, n)):
            c    = float(np.corrcoef(
                self.binary[lag:], self.binary[:-lag]
            )[0, 1])
            is_s = bool(abs(c) > bound)
            acfs.append({
                'lag'            : int(lag),
                'autocorrelation': round(c, 6),
                'significant'    : is_s,
                'bound'          : round(bound, 4)
            })
            if is_s:
                sig.append(int(lag))
        r = {
            'test_name'         : 'Autocorrelation Test',
            'significance_bound': round(bound, 4),
            'significant_lags'  : sig,
            'n_significant'     : int(len(sig)),
            'autocorrelations'  : acfs,
            'p_value'           : round(0.003 if sig else 0.5, 4),
            'passed'            : bool(len(sig) == 0),
            'interpretation'    : (
                '✅ لا ارتباط — عشوائي'
                if not sig
                else f'🔴 ارتباط دال في Lags: {sig}'
            )
        }
        self.results['autocorrelation'] = r
        return r

    def distribution_test(self) -> dict:
        bins   = [1.0,1.5,2.0,3.0,5.0,10.0,float('inf')]
        labels = ['1-1.5x','1.5-2x','2-3x','3-5x','5-10x','>10x']
        n      = int(len(self.raw))
        he     = 0.99
        th     = []
        for i in range(len(bins)-1):
            pl = float(min(he/bins[i], 1.0))
            ph = float(
                min(he/bins[i+1], 1.0)
                if bins[i+1] != float('inf') else 0.0
            )
            th.append(pl - ph)
        s  = sum(th)
        th = [p/s for p in th]
        obs = []
        for i in range(len(bins)-1):
            if bins[i+1] == float('inf'):
                obs.append(int((self.raw >= bins[i]).sum()))
            else:
                obs.append(int(
                    ((self.raw >= bins[i]) &
                     (self.raw < bins[i+1])).sum()
                ))
        exp = [float(p*n) for p in th]
        try:
            c2, p = stats.chisquare(obs, exp)
            c2 = float(c2); p = float(p)
        except Exception:
            c2, p = 0.0, 1.0
        r = {
            'test_name'     : 'Chi-Square Distribution Test',
            'categories'    : [
                {
                    'range'       : labels[i],
                    'observed'    : int(obs[i]),
                    'expected'    : round(exp[i], 1),
                    'observed_pct': round(obs[i]/n*100, 1),
                    'expected_pct': round(th[i]*100, 1)
                }
                for i in range(len(labels))
            ],
            'chi2_statistic': round(c2, 4),
            'p_value'       : round(p, 4),
            'passed'        : bool(p >= 0.01),
            'interpretation': (
                '✅ يتبع Power Law'
                if p >= 0.01 else '🔴 انحراف عن التوزيع!'
            )
        }
        self.results['distribution'] = r
        return r

    def entropy_test(self) -> dict:
        ph  = float(self.binary.mean())
        pl  = float(1 - ph)
        ent = float(
            -(ph*np.log2(ph) + pl*np.log2(pl))
            if ph > 0 and pl > 0 else 0.0
        )
        r = {
            'test_name'     : 'Shannon Entropy Test',
            'binary_entropy': round(ent, 6),
            'efficiency_pct': round(ent*100, 2),
            'p_value'       : round(0.95 if ent >= 0.95 else 0.001, 4),
            'passed'        : bool(ent >= 0.95),
            'interpretation': (
                f'✅ إنتروبيا عالية ({ent:.3f} bits)'
                if ent >= 0.95
                else f'🔴 إنتروبيا منخفضة ({ent:.3f})!'
            )
        }
        self.results['entropy'] = r
        return r

    def serial_test(self) -> dict:
        pairs   = Counter(zip(self.binary[:-1], self.binary[1:]))
        n_pairs = int(len(self.binary) - 1)
        exp     = float(n_pairs / 4)
        obs     = [
            int(pairs.get((0,0), 0)),
            int(pairs.get((0,1), 0)),
            int(pairs.get((1,0), 0)),
            int(pairs.get((1,1), 0))
        ]
        c2 = float(sum((o-exp)**2/exp for o in obs))
        p  = float(1 - chi2.cdf(c2, df=3))
        r  = {
            'test_name'    : 'Serial (Pairs) Test',
            'pair_counts'  : {
                'L→L': int(pairs.get((0,0),0)),
                'L→H': int(pairs.get((0,1),0)),
                'H→L': int(pairs.get((1,0),0)),
                'H→H': int(pairs.get((1,1),0))
            },
            'expected_each'  : round(exp, 1),
            'chi2_statistic' : round(c2, 4),
            'p_value'        : round(p, 4),
            'passed'         : bool(p >= 0.01),
            'interpretation' : (
                '✅ توزيع الأزواج عشوائي'
                if p >= 0.01 else '🔴 أنماط في الأزواج!'
            )
        }
        self.results['serial'] = r
        return r

    def longest_run_test(self) -> dict:
        mh = ml = ch = cl = 0
        for v in self.binary:
            if v == 1:
                ch += 1; cl = 0; mh = max(mh, ch)
            else:
                cl += 1; ch = 0; ml = max(ml, cl)
        n   = int(len(self.binary))
        thr = float(3 * np.log2(n)) if n > 1 else 10.0
        ok  = bool(mh <= thr * 1.5)
        r   = {
            'test_name'           : 'Longest Run Test',
            'max_consecutive_high': int(mh),
            'max_consecutive_low' : int(ml),
            'theoretical_max'     : round(thr, 1),
            'p_value'             : round(0.1 if ok else 0.001, 4),
            'passed'              : ok,
            'interpretation'      : (
                f'✅ أطول تسلسل ({mh}) ضمن الحدود'
                if ok else f'🔴 تسلسل غير طبيعي: {mh}!'
            )
        }
        self.results['longest_run'] = r
        return r

    def run_all(self) -> dict:
        fns = [
            self.frequency_test,
            self.runs_test,
            self.autocorrelation_test,
            self.distribution_test,
            self.entropy_test,
            self.serial_test,
            self.longest_run_test,
        ]
        passed = 0
        for fn in fns:
            if fn().get('passed', False):
                passed += 1
        total   = int(len(fns))
        verdict = (
            'عشوائي إحصائياً ✅'
            if passed >= total * 0.75
            else 'يحتوي أنماط إحصائية 🔴'
        )
        return {
            'passed_tests': int(passed),
            'total_tests' : total,
            'verdict'     : verdict
        }


# ══════════════════════════════════════════════════════════════
#         2. محرك كشف الأنماط المتقدم
# ══════════════════════════════════════════════════════════════
class PatternDetector:
    """
    يكتشف الأنماط المتكررة في تسلسل Crash:
    - أنماط N→M (بعد N منخفض يأتي M)
    - تسلسلات متكررة
    - دوريات زمنية
    - أنماط Markov من الدرجة 1 و 2 و 3
    """

    def __init__(self, data: list):
        self.raw  = np.array(data, dtype=float)
        self.cats = [categorize(v) for v in data]
        self.bin  = [1 if v >= 2.0 else 0 for v in data]
        self.n    = len(data)

    # ── Markov من الدرجة 1 ──────────────────────────────────
    def markov_order1(self) -> dict:
        """
        مصفوفة الانتقال: بعد كل فئة ماذا يأتي؟
        """
        trans  = defaultdict(Counter)
        for i in range(len(self.cats) - 1):
            trans[self.cats[i]][self.cats[i+1]] += 1

        matrix = {}
        for cat in CAT_ORDER:
            total = sum(trans[cat].values())
            if total > 0:
                matrix[cat] = {
                    c: round(trans[cat][c] / total, 4)
                    for c in CAT_ORDER
                }
            else:
                matrix[cat] = {c: 0.0 for c in CAT_ORDER}

        # أفضل انتقال لكل حالة
        best_transitions = {}
        for cat in CAT_ORDER:
            if any(matrix[cat].values()):
                best_next = max(matrix[cat], key=matrix[cat].get)
                best_prob = matrix[cat][best_next]
                best_transitions[cat] = {
                    'next'       : best_next,
                    'probability': best_prob,
                    'label'      : CAT_LABELS[best_next]
                }

        return {
            'matrix'          : matrix,
            'best_transitions': best_transitions
        }

    # ── Markov من الدرجة 2 ──────────────────────────────────
    def markov_order2(self) -> dict:
        """
        بعد كل زوج (A,B) ماذا يأتي؟
        """
        trans = defaultdict(Counter)
        for i in range(len(self.cats) - 2):
            key = (self.cats[i], self.cats[i+1])
            trans[key][self.cats[i+2]] += 1

        # أقوى الأنماط
        patterns = []
        for (a, b), counts in trans.items():
            total = sum(counts.values())
            if total >= 3:
                best  = max(counts, key=counts.get)
                prob  = counts[best] / total
                if prob >= 0.55:
                    patterns.append({
                        'pattern'    : f"{CAT_LABELS[a]} → {CAT_LABELS[b]}",
                        'next'       : CAT_LABELS[best],
                        'probability': round(float(prob), 4),
                        'occurrences': int(total)
                    })

        patterns.sort(key=lambda x: x['probability'], reverse=True)
        return {'top_patterns': patterns[:10]}

    # ── Markov من الدرجة 3 ──────────────────────────────────
    def markov_order3(self) -> dict:
        """
        بعد ثلاث قيم متتالية ماذا يأتي؟
        """
        trans = defaultdict(Counter)
        for i in range(len(self.cats) - 3):
            key = (self.cats[i], self.cats[i+1], self.cats[i+2])
            trans[key][self.cats[i+3]] += 1

        patterns = []
        for (a, b, c), counts in trans.items():
            total = sum(counts.values())
            if total >= 3:
                best = max(counts, key=counts.get)
                prob = counts[best] / total
                if prob >= 0.60:
                    patterns.append({
                        'pattern': (
                            f"{CAT_LABELS[a]} → "
                            f"{CAT_LABELS[b]} → "
                            f"{CAT_LABELS[c]}"
                        ),
                        'next'       : CAT_LABELS[best],
                        'probability': round(float(prob), 4),
                        'occurrences': int(total)
                    })

        patterns.sort(key=lambda x: x['probability'], reverse=True)
        return {'top_patterns': patterns[:8]}

    # ── تحليل التسلسل المنخفض ───────────────────────────────
    def streak_analysis(self) -> dict:
        """
        بعد k قيم منخفضة متتالية ما احتمال القادمة؟
        """
        results = []
        for k in range(1, 8):
            transitions = 0
            occurrences = 0
            for i in range(len(self.bin) - k):
                if all(self.bin[i:i+k] == np.zeros(k)):
                    occurrences += 1
                    if i + k < len(self.bin):
                        if self.bin[i+k] == 1:
                            transitions += 1
            if occurrences >= 3:
                prob = (transitions + 1) / (occurrences + 2)
                results.append({
                    'streak_length'        : int(k),
                    'occurrences'          : int(occurrences),
                    'times_followed_by_high': int(transitions),
                    'prob_next_high'       : round(float(prob), 4),
                    'prob_pct'             : round(float(prob)*100, 1)
                })
        return {'streak_results': results}

    # ── اكتشاف الدوريات بـ FFT ──────────────────────────────
    def fft_cycles(self) -> dict:
        """
        تحليل طيفي لاكتشاف الدوريات الزمنية
        """
        n        = len(self.raw)
        log_data = np.log(np.maximum(self.raw, 1.0))
        centered = log_data - log_data.mean()
        mag      = np.abs(fft(centered))
        freqs    = fftfreq(n)
        half     = n // 2
        mag_h    = mag[:half]
        freq_h   = freqs[:half]

        # اكتشاف القمم
        peaks, props = find_peaks(
            mag_h,
            height=mag_h.mean() * 2,
            distance=3
        )

        cycles = []
        for pk in peaks:
            if float(freq_h[pk]) > 0:
                period = float(1 / freq_h[pk])
                cycles.append({
                    'period_rounds': round(period, 1),
                    'magnitude'    : round(float(mag_h[pk]), 4),
                    'relative_power': round(
                        float(mag_h[pk]) / (mag_h.max() + 1e-9), 4
                    )
                })

        cycles.sort(key=lambda x: x['relative_power'], reverse=True)
        dom = float(mag_h.max() / (mag_h.mean() + 1e-9))

        return {
            'detected_cycles'  : cycles[:6],
            'dominance_ratio'  : round(dom, 2),
            'has_strong_cycle' : bool(dom > 8),
            'n_peaks'          : int(len(peaks))
        }

    # ── أنماط الأرقام الكبيرة ───────────────────────────────
    def big_number_patterns(self) -> dict:
        """
        كم جولة تمر بين كل ارتفاع كبير؟
        """
        thresholds = [2.0, 3.0, 5.0, 10.0]
        results    = {}

        for thr in thresholds:
            positions = [
                i for i, v in enumerate(self.raw)
                if v >= thr
            ]
            if len(positions) >= 2:
                gaps = [
                    int(positions[i+1] - positions[i])
                    for i in range(len(positions)-1)
                ]
                results[f'>={thr}x'] = {
                    'count'       : int(len(positions)),
                    'avg_gap'     : round(float(np.mean(gaps)), 1),
                    'std_gap'     : round(float(np.std(gaps)), 1),
                    'min_gap'     : int(np.min(gaps)),
                    'max_gap'     : int(np.max(gaps)),
                    'median_gap'  : round(float(np.median(gaps)), 1),
                    'last_seen'   : int(len(self.raw) - 1 - positions[-1]),
                    'gap_history' : gaps[-20:]
                }

        return results

    # ── تحليل النافذة الزمنية ───────────────────────────────
    def sliding_window_analysis(self, window: int = 20) -> dict:
        """
        كيف تتغير احتمالات الفئات عبر الوقت؟
        """
        records = []
        for i in range(0, len(self.raw) - window, window // 2):
            seg  = self.raw[i:i+window]
            high = float(np.mean(seg >= 2.0))
            avg  = float(np.mean(seg))
            records.append({
                'window_start': int(i),
                'window_end'  : int(i + window),
                'pct_high'    : round(high * 100, 1),
                'avg_value'   : round(avg, 2),
                'max_value'   : round(float(np.max(seg)), 2)
            })
        return {'windows': records}

    def run_all(self) -> dict:
        return {
            'markov1'       : self.markov_order1(),
            'markov2'       : self.markov_order2(),
            'markov3'       : self.markov_order3(),
            'streaks'       : self.streak_analysis(),
            'fft_cycles'    : self.fft_cycles(),
            'big_numbers'   : self.big_number_patterns(),
            'sliding_window': self.sliding_window_analysis()
        }


# ══════════════════════════════════════════════════════════════
#         3. محرك التنبؤ المتقدم
# ══════════════════════════════════════════════════════════════
class AdvancedPredictor:
    """
    يدمج عدة طرق للتنبؤ بالقيمة القادمة:
    1. Markov Chain (درجة 1، 2، 3)
    2. تحليل التسلسل الحالي
    3. الموقع في الدورة الزمنية
    4. الانحراف عن المتوسط التاريخي
    5. التصويت الجماعي للطرق
    """

    def __init__(self, data: list, pattern_results: dict):
        self.raw      = np.array(data, dtype=float)
        self.cats     = [categorize(v) for v in data]
        self.bin      = [1 if v >= 2.0 else 0 for v in data]
        self.patterns = pattern_results
        self.n        = len(data)

    def _current_state(self) -> dict:
        """الحالة الراهنة للتسلسل"""
        last1 = self.cats[-1]
        last2 = self.cats[-2] if self.n >= 2 else last1
        last3 = self.cats[-3] if self.n >= 3 else last1

        # التسلسل المنخفض الحالي
        low_streak = 0
        for v in reversed(self.bin):
            if v == 0:
                low_streak += 1
            else:
                break

        # التسلسل المرتفع الحالي
        high_streak = 0
        for v in reversed(self.bin):
            if v == 1:
                high_streak += 1
            else:
                break

        return {
            'last1'      : last1,
            'last2'      : last2,
            'last3'      : last3,
            'last_value' : float(self.raw[-1]),
            'low_streak' : int(low_streak),
            'high_streak': int(high_streak),
            'recent_avg' : float(np.mean(self.raw[-10:])),
            'hist_avg'   : float(np.mean(self.raw))
        }

    def predict_markov1(self, state: dict) -> dict:
        """توقع Markov الدرجة 1"""
        matrix = self.patterns['markov1']['matrix']
        probs  = matrix.get(state['last1'], {})
        if not probs or sum(probs.values()) == 0:
            return {'category': 'MED', 'confidence': 0.33,
                    'method': 'Markov-1'}
        best = max(probs, key=probs.get)
        return {
            'category'  : best,
            'confidence': float(probs[best]),
            'all_probs' : probs,
            'method'    : 'Markov-1'
        }

    def predict_markov2(self, state: dict) -> dict:
        """توقع Markov الدرجة 2"""
        patterns = self.patterns['markov2']['top_patterns']
        key = f"{CAT_LABELS[state['last2']]} → {CAT_LABELS[state['last1']]}"
        for p in patterns:
            if p['pattern'] == key:
                # العثور على فئة من الـ label
                for cat, lbl in CAT_LABELS.items():
                    if lbl == p['next']:
                        return {
                            'category'  : cat,
                            'confidence': float(p['probability']),
                            'method'    : 'Markov-2'
                        }
        return {'category': 'MED', 'confidence': 0.33,
                'method': 'Markov-2'}

    def predict_streak(self, state: dict) -> dict:
        """توقع بناءً على التسلسل الحالي"""
        streak = state['low_streak']
        streaks = self.patterns['streaks']['streak_results']

        for s in streaks:
            if s['streak_length'] == streak:
                prob = float(s['prob_next_high'])
                cat  = 'MED' if prob >= 0.55 else 'LOW'
                return {
                    'category'       : cat,
                    'confidence'     : prob if prob >= 0.5 else (1-prob),
                    'prob_high'      : prob,
                    'streak_length'  : streak,
                    'method'         : 'Streak Analysis'
                }

        # إذا لم يوجد — بناءً على المتوسط التاريخي
        hist_prob = float(np.mean(self.bin))
        return {
            'category'  : 'MED' if hist_prob >= 0.5 else 'LOW',
            'confidence': hist_prob,
            'method'    : 'Streak Analysis (default)'
        }

    def predict_cycle_position(self) -> dict:
        """توقع بناءً على الموقع في الدورة"""
        cycles = self.patterns['fft_cycles']['detected_cycles']
        if not cycles:
            return {
                'category'  : 'MED',
                'confidence': 0.4,
                'method'    : 'Cycle Position'
            }

        best_cycle = cycles[0]
        period     = float(best_cycle['period_rounds'])
        position   = (self.n % max(int(period), 1))
        phase      = position / max(period, 1)

        # المرحلة: 0-0.3 = صعود، 0.3-0.7 = ذروة، 0.7-1.0 = هبوط
        if phase < 0.3:
            cat  = 'HIGH'
            conf = 0.55
        elif phase < 0.7:
            cat  = 'VERY_HIGH'
            conf = 0.50
        else:
            cat  = 'LOW'
            conf = 0.55

        return {
            'category'     : cat,
            'confidence'   : conf,
            'cycle_period' : round(period, 1),
            'phase'        : round(float(phase), 3),
            'method'       : 'Cycle Position'
        }

    def predict_mean_reversion(self, state: dict) -> dict:
        """
        نظرية الانتقال للمتوسط:
        إذا كان المتوسط الأخير أقل من التاريخي ← توقع ارتفاع
        """
        recent = float(state['recent_avg'])
        hist   = float(state['hist_avg'])
        ratio  = recent / (hist + 1e-9)

        if ratio < 0.75:
            cat  = 'HIGH'
            conf = min(float(1 - ratio + 0.5), 0.75)
        elif ratio < 0.90:
            cat  = 'MED'
            conf = 0.55
        elif ratio > 1.25:
            cat  = 'VERY_LOW'
            conf = min(float(ratio - 0.75), 0.70)
        else:
            cat  = 'MED'
            conf = 0.45

        return {
            'category'    : cat,
            'confidence'  : round(conf, 4),
            'recent_avg'  : round(recent, 2),
            'hist_avg'    : round(hist, 2),
            'ratio'       : round(float(ratio), 4),
            'method'      : 'Mean Reversion'
        }

    def predict_big_number_due(self, state: dict) -> dict:
        """
        كم جولة مضت منذ آخر ارتفاع كبير؟
        هل حان وقته؟
        """
        big = self.patterns['big_numbers']
        predictions = {}

        for key, info in big.items():
            last_seen = info['last_seen']
            avg_gap   = info['avg_gap']
            std_gap   = info['std_gap']

            # نسبة الاستحقاق
            due_ratio = last_seen / (avg_gap + 1e-9)

            if due_ratio >= 1.5:
                urgency = 'مرتفع جداً 🔴'
                conf    = min(0.75, float(due_ratio) / 3)
            elif due_ratio >= 1.0:
                urgency = 'مرتفع ⚠️'
                conf    = 0.60
            elif due_ratio >= 0.7:
                urgency = 'متوسط 🟡'
                conf    = 0.50
            else:
                urgency = 'منخفض ✅'
                conf    = 0.35

            predictions[key] = {
                'last_seen' : int(last_seen),
                'avg_gap'   : float(avg_gap),
                'due_ratio' : round(float(due_ratio), 2),
                'urgency'   : urgency,
                'confidence': round(conf, 3)
            }

        return {
            'predictions': predictions,
            'method'     : 'Big Number Due Analysis'
        }

    def ensemble_predict(self) -> dict:
        """
        التنبؤ الجماعي: يجمع كل الطرق بأوزان
        """
        state = self._current_state()

        # تشغيل كل طريقة
        m1  = self.predict_markov1(state)
        m2  = self.predict_markov2(state)
        str_= self.predict_streak(state)
        cyc = self.predict_cycle_position()
        rev = self.predict_mean_reversion(state)

        methods = [m1, m2, str_, cyc, rev]
        weights = [0.25, 0.20, 0.25, 0.15, 0.15]

        # التصويت المُرجَّح
        votes = defaultdict(float)
        for method, w in zip(methods, weights):
            cat   = method['category']
            conf  = float(method.get('confidence', 0.5))
            votes[cat] += w * conf

        # الفائز
        winner = max(votes, key=votes.get)
        total  = sum(votes.values())
        final_conf = float(votes[winner]) / total if total > 0 else 0.5

        # احتمال الارتفاع (>= 2x)
        high_cats  = ['MED','HIGH','VERY_HIGH','EXTREME']
        prob_high  = sum(
            float(votes[c]) for c in high_cats
        ) / (total + 1e-9)

        # تعديل بناءً على التسلسل المنخفض
        low_streak = state['low_streak']
        if low_streak >= 4:
            boost    = min(0.10 * (low_streak - 3), 0.20)
            prob_high = min(float(prob_high) + boost, 0.90)

        # الفئة المتوقعة النهائية
        if prob_high >= 0.70:
            final_cat = 'HIGH'
        elif prob_high >= 0.55:
            final_cat = 'MED'
        elif prob_high <= 0.30:
            final_cat = 'VERY_LOW'
        else:
            final_cat = winner

        # تقدير القيمة العددية
        cat_ranges = {
            'VERY_LOW' : (1.0, 1.5),
            'LOW'      : (1.5, 2.0),
            'MED'      : (2.0, 3.0),
            'HIGH'     : (3.0, 5.0),
            'VERY_HIGH': (5.0, 10.0),
            'EXTREME'  : (10.0, 30.0)
        }
        lo, hi     = cat_ranges[final_cat]
        est_value  = float((lo + hi) / 2)

        # مستوى الثقة الكلي
        confidence_level = (
            'عالية 🟢'   if final_conf >= 0.65 else
            'متوسطة 🟡'  if final_conf >= 0.50 else
            'منخفضة 🔴'
        )

        return {
            'final_category'  : final_cat,
            'final_label'     : CAT_LABELS[final_cat],
            'estimated_value' : round(est_value, 2),
            'value_range'     : f"{lo:.1f}x — {hi:.1f}x",
            'prob_high'       : round(float(prob_high), 4),
            'prob_high_pct'   : round(float(prob_high)*100, 1),
            'confidence'      : round(final_conf, 4),
            'confidence_level': confidence_level,
            'low_streak'      : int(low_streak),
            'current_state'   : state,
            'method_votes'    : {
                k: round(float(v), 4)
                for k, v in sorted(
                    votes.items(),
                    key=lambda x: x[1], reverse=True
                )
            },
            'individual_methods': {
                'markov1'      : m1,
                'markov2'      : m2,
                'streak'       : str_,
                'cycle'        : cyc,
                'mean_reversion': rev
            },
            'big_number_due'  : self.predict_big_number_due(state)
        }


# ══════════════════════════════════════════════════════════════
#                    بيانات نموذجية
# ══════════════════════════════════════════════════════════════
SAMPLE_DATA = [
    8.72,6.75,1.86,2.18,1.25,2.28,1.24,1.20,1.54,24.46,
    4.16,1.49,1.09,1.47,1.54,1.53,2.10,32.04,11.0,1.17,
    1.70,2.61,1.26,22.23,1.77,1.93,3.35,7.01,1.83,9.39,
    3.31,2.04,1.30,6.65,1.16,3.39,1.95,10.85,1.65,1.22,
    1.60,4.67,1.85,2.72,1.00,3.02,1.35,1.30,1.37,17.54,
    1.18,1.00,14.40,1.11,6.15,2.39,2.22,1.42,1.23,2.42,
    1.07,1.24,2.55,7.26,1.69,5.10,2.59,5.51,2.31,2.12,
    1.97,1.50,3.01,2.29,1.36,4.95,5.09,8.50,1.77,5.52,
    3.93,1.50,2.28,2.49,18.25,1.68,1.42,2.12,4.17,1.04,
    2.35,1.00,1.01,5.46,1.13,2.84,3.39,2.79,1.59,1.53,
    4.34,2.96,1.06,1.72,2.16,2.20,3.61,2.34,4.49,1.72,
    1.78,9.27,8.49,2.86,1.66,4.63,9.25,1.35,1.00,1.64,
    1.86,2.81,2.44,1.74,1.10,1.29,1.45,8.92,1.24,6.39,
    1.16,1.19,2.40,4.64,3.17,24.21,1.17,1.42,2.13,1.12,
    3.78,1.12,1.52,22.81,1.31,1.90,1.38,1.47,2.86,1.79,
]


# ══════════════════════════════════════════════════════════════
#                    واجهة المستخدم
# ══════════════════════════════════════════════════════════════
st.title("🎓 محلل PRNG والتنبؤ بالأنماط")
st.caption(
    "اختبارات NIST + كشف أنماط Markov + "
    "تحليل دورات FFT + تنبؤ جماعي | مشروع تخرج"
)

# ── إدخال البيانات ──────────────────────────────────────────
st.header("📥 إدخال البيانات")
method = st.radio(
    "طريقة الإدخال:",
    ["📝 يدوي", "📂 CSV", "🎲 نموذجية"],
    horizontal=True
)

raw_data = None

if method == "📝 يدوي":
    txt = st.text_area(
        "أدخل قيم Crash (30 على الأقل):",
        height=120,
        placeholder="1.23  4.56  2.10  8.92  3.41  22.3 ..."
    )
    if txt.strip():
        try:
            raw_data = [
                float(x)
                for x in txt.replace('\n',' ').split()
                if x.strip()
            ]
            st.success(f"✅ {len(raw_data)} قيمة")
        except Exception:
            st.error("❌ أرقام فقط مفصولة بمسافات")

elif method == "📂 CSV":
    up = st.file_uploader("CSV مع عمود crash_point", type=['csv'])
    if up:
        try:
            df_up = pd.read_csv(up)
            if 'crash_point' in df_up.columns:
                raw_data = [float(x) for x in
                            df_up['crash_point'].dropna()]
                st.success(f"✅ {len(raw_data)} قيمة")
            else:
                st.error(f"الأعمدة: {list(df_up.columns)}")
        except Exception as e:
            st.error(str(e))
else:
    raw_data = SAMPLE_DATA
    st.info(f"🎲 {len(raw_data)} قيمة نموذجية")

# ── ملخص سريع ───────────────────────────────────────────────
if raw_data:
    arr = np.array(raw_data, dtype=float)
    n   = int(len(arr))

    st.markdown("---")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("العدد",      str(n))
    c2.metric("المتوسط",    f"{arr.mean():.2f}x")
    c3.metric("الوسيط",     f"{np.median(arr):.2f}x")
    c4.metric("الأقصى",     f"{arr.max():.2f}x")
    c5.metric(">= 2x",      f"{np.mean(arr>=2)*100:.1f}%")
    c6.metric("آخر قيمة",   f"{arr[-1]:.2f}x")

    if n < 50:
        st.warning(f"⚠️ يُفضَّل 50+ قيمة (لديك {n})")
    else:
        st.markdown("---")
        if st.button(
            "🚀 تشغيل التحليل الكامل والتنبؤ",
            type="primary",
            use_container_width=True
        ):
            prog   = st.progress(0)
            status = st.empty()

            # اختبارات NIST
            status.info("⏳ اختبارات NIST...")
            suite    = RandomnessTestSuite(raw_data)
            nist_res = suite.run_all()
            prog.progress(25)

            # كشف الأنماط
            status.info("⏳ كشف الأنماط...")
            detector = PatternDetector(raw_data)
            pat_res  = detector.run_all()
            prog.progress(60)

            # التنبؤ
            status.info("⏳ حساب التنبؤ الجماعي...")
            predictor = AdvancedPredictor(raw_data, pat_res)
            pred      = predictor.ensemble_predict()
            prog.progress(100)

            status.empty()
            prog.empty()
            st.balloons()

            # ════════════════════════════════════════════════
            #               عرض النتائج
            # ════════════════════════════════════════════════

            # ── 1. التنبؤ الرئيسي ────────────────────────
            st.markdown("---")
            st.header("🎯 التنبؤ بالجولة القادمة")

            cat   = pred['final_category']
            color = CAT_COLORS.get(cat, '#888')

            col_pred, col_info = st.columns([1, 2])

            with col_pred:
                st.markdown(
                    f"""
                    <div style="
                        background:{color}22;
                        border: 3px solid {color};
                        border-radius: 16px;
                        padding: 24px;
                        text-align: center;
                    ">
                    <h2 style="color:{color}; margin:0;">
                        {pred['final_label']}
                    </h2>
                    <h3 style="margin:8px 0;">
                        ≈ {pred['estimated_value']}x
                    </h3>
                    <p style="margin:0; font-size:0.9em;">
                        نطاق: {pred['value_range']}
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_info:
                ca,cb,cc,cd = st.columns(4)
                ca.metric(
                    "احتمال >= 2x",
                    f"{pred['prob_high_pct']}%"
                )
                cb.metric(
                    "الثقة",
                    f"{pred['confidence']*100:.1f}%",
                    delta=pred['confidence_level']
                )
                cc.metric(
                    "تسلسل منخفض",
                    f"{pred['low_streak']} جولة"
                )
                cd.metric(
                    "المتوسط الأخير",
                    f"{pred['current_state']['recent_avg']:.2f}x"
                )

                # أصوات الطرق
                st.markdown("**أصوات طرق التنبؤ:**")
                votes_df = pd.DataFrame({
                    'الفئة': [CAT_LABELS.get(k,k)
                              for k in pred['method_votes'].keys()],
                    'الوزن': list(pred['method_votes'].values())
                })
                fig_votes = px.bar(
                    votes_df, x='الفئة', y='الوزن',
                    color='الوزن',
                    color_continuous_scale='RdYlGn',
                    title="توزيع أصوات الطرق"
                )
                fig_votes.update_layout(height=220, margin=dict(t=30))
                st.plotly_chart(fig_votes, use_container_width=True)

            # ── 2. تفاصيل الطرق ──────────────────────────
            st.markdown("---")
            st.subheader("🔍 تفاصيل طرق التنبؤ")

            methods_data = []
            im = pred['individual_methods']
            for key, label in [
                ('markov1',       'Markov درجة 1'),
                ('markov2',       'Markov درجة 2'),
                ('streak',        'تحليل التسلسل'),
                ('cycle',         'موقع الدورة'),
                ('mean_reversion','الانتقال للمتوسط')
            ]:
                m = im[key]
                methods_data.append({
                    'الطريقة'  : label,
                    'التوقع'   : CAT_LABELS.get(m['category'], m['category']),
                    'الثقة %'  : f"{m.get('confidence', 0)*100:.1f}%"
                })
            st.dataframe(
                pd.DataFrame(methods_data),
                use_container_width=True,
                hide_index=True
            )

            # ── 3. تحليل استحقاق الأرقام الكبيرة ─────────
            st.subheader("⏰ استحقاق الأرقام الكبيرة")
            due = pred['big_number_due']['predictions']
            due_rows = []
            for thr, info in due.items():
                due_rows.append({
                    'العتبة'       : thr,
                    'آخر ظهور'     : f"منذ {info['last_seen']} جولة",
                    'متوسط الفجوة' : f"{info['avg_gap']:.1f}",
                    'نسبة الاستحقاق': f"{info['due_ratio']}x",
                    'الإلحاحية'    : info['urgency'],
                    'الثقة'        : f"{info['confidence']*100:.0f}%"
                })
            st.dataframe(
                pd.DataFrame(due_rows),
                use_container_width=True,
                hide_index=True
            )

            # ════════════════════════════════════════════════
            #            الأنماط المكتشفة
            # ════════════════════════════════════════════════
            st.markdown("---")
            st.header("🔬 الأنماط المكتشفة")

            tab_mk1, tab_mk2, tab_mk3, tab_str, \
            tab_fft, tab_win, tab_nist = st.tabs([
                "🔗 Markov-1",
                "🔗 Markov-2",
                "🔗 Markov-3",
                "📊 التسلسل",
                "📡 الدورات",
                "📈 النافذة",
                "🧪 NIST"
            ])

            # Markov-1: مصفوفة الانتقال
            with tab_mk1:
                st.subheader("مصفوفة الانتقال (Markov درجة 1)")
                matrix = pat_res['markov1']['matrix']
                df_mat = pd.DataFrame(matrix).T
                df_mat.index.name   = 'من \\ إلى'
                df_mat.columns.name = None
                df_mat = df_mat.rename(
                    index=CAT_LABELS,
                    columns=CAT_LABELS
                )

                fig_heat = px.imshow(
                    df_mat.values,
                    x=list(df_mat.columns),
                    y=list(df_mat.index),
                    color_continuous_scale='RdYlGn',
                    title="مصفوفة الانتقال — كل خلية = P(العمود | الصف)",
                    zmin=0, zmax=1,
                    text_auto='.2f'
                )
                fig_heat.update_layout(height=420)
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption(
                    "القراءة: الصف = الحالة الحالية، "
                    "العمود = الحالة التالية. "
                    "اللون الأخضر = احتمال عالٍ."
                )

                st.subheader("أفضل انتقال لكل حالة")
                bt = pat_res['markov1']['best_transitions']
                bt_rows = []
                for cat, info in bt.items():
                    bt_rows.append({
                        'الحالة الحالية': CAT_LABELS.get(cat, cat),
                        'التوقع التالي' : info['label'],
                        'الاحتمال'      : f"{info['probability']*100:.1f}%"
                    })
                st.dataframe(
                    pd.DataFrame(bt_rows),
                    use_container_width=True,
                    hide_index=True
                )

            # Markov-2
            with tab_mk2:
                st.subheader("أنماط Markov درجة 2 (الأقوى)")
                p2 = pat_res['markov2']['top_patterns']
                if p2:
                    df_p2 = pd.DataFrame(p2)
                    df_p2['probability'] = df_p2['probability'].apply(
                        lambda x: f"{x*100:.1f}%"
                    )
                    df_p2.columns = [
                        'النمط','التوقع التالي',
                        'الاحتمال','عدد المرات'
                    ]
                    st.dataframe(
                        df_p2, use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("لا توجد أنماط بدرجة ثقة كافية")

            # Markov-3
            with tab_mk3:
                st.subheader("أنماط Markov درجة 3 (الأقوى)")
                p3 = pat_res['markov3']['top_patterns']
                if p3:
                    df_p3 = pd.DataFrame(p3)
                    df_p3['probability'] = df_p3['probability'].apply(
                        lambda x: f"{x*100:.1f}%"
                    )
                    df_p3.columns = [
                        'النمط','التوقع التالي',
                        'الاحتمال','عدد المرات'
                    ]
                    st.dataframe(
                        df_p3, use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("لا توجد أنماط بدرجة ثقة >= 60%")

            # التسلسل
            with tab_str:
                st.subheader(
                    "احتمال الارتفاع بعد k جولة منخفضة متتالية"
                )
                sr = pat_res['streaks']['streak_results']
                if sr:
                    df_sr = pd.DataFrame(sr)
                    fig_sr = px.bar(
                        df_sr,
                        x='streak_length',
                        y='prob_next_high',
                        title="P(القادم >= 2x | k جولة منخفضة متتالية)",
                        labels={
                            'streak_length' : 'طول التسلسل المنخفض',
                            'prob_next_high': 'الاحتمال'
                        },
                        color='prob_next_high',
                        color_continuous_scale='RdYlGn',
                        text='prob_pct'
                    )
                    fig_sr.update_traces(
                        texttemplate='%{text}%',
                        textposition='outside'
                    )
                    fig_sr.add_hline(
                        y=0.5, line_dash="dash",
                        line_color="blue",
                        annotation_text="50% (خط الحياد)"
                    )
                    fig_sr.update_layout(height=420)
                    st.plotly_chart(fig_sr, use_container_width=True)

                    df_sr2 = df_sr.copy()
                    df_sr2['prob_next_high'] = df_sr2[
                        'prob_next_high'
                    ].apply(lambda x: f"{x*100:.1f}%")
                    df_sr2.columns = [
                        'طول التسلسل','عدد الحالات',
                        'تبعه ارتفاع','الاحتمال','%'
                    ]
                    st.dataframe(
                        df_sr2, use_container_width=True,
                        hide_index=True
                    )

            # الدورات
            with tab_fft:
                st.subheader("الدورات الزمنية المكتشفة (FFT)")
                fc = pat_res['fft_cycles']
                ca, cb = st.columns(2)
                ca.metric(
                    "نسبة الهيمنة",
                    f"{fc['dominance_ratio']}x"
                )
                cb.metric(
                    "دورة قوية؟",
                    "نعم 🔴" if fc['has_strong_cycle'] else "لا ✅"
                )

                if fc['detected_cycles']:
                    df_fc = pd.DataFrame(fc['detected_cycles'])
                    fig_fc = px.bar(
                        df_fc,
                        x='period_rounds',
                        y='relative_power',
                        title="الدورات الزمنية المكتشفة",
                        labels={
                            'period_rounds' : 'دورة (جولات)',
                            'relative_power': 'القوة النسبية'
                        },
                        color='relative_power',
                        color_continuous_scale='Reds',
                        text='period_rounds'
                    )
                    fig_fc.update_traces(
                        texttemplate='%{text:.0f}j',
                        textposition='outside'
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.dataframe(
                        df_fc, use_container_width=True,
                        hide_index=True
                    )

                # فجوات الأرقام الكبيرة
                st.subheader("تاريخ فجوات الأرقام الكبيرة")
                bn = pat_res['big_numbers']
                for key, info in bn.items():
                    with st.expander(f"📊 {key}"):
                        ca2, cb2, cc2, cd2 = st.columns(4)
                        ca2.metric("عدد الظهورات", info['count'])
                        cb2.metric("متوسط الفجوة",
                                   f"{info['avg_gap']:.1f}")
                        cc2.metric("أدنى فجوة", info['min_gap'])
                        cd2.metric("أعلى فجوة", info['max_gap'])

                        if info['gap_history']:
                            fig_gap = px.line(
                                y=info['gap_history'],
                                title=f"تاريخ الفجوات — {key}",
                                labels={'y':'الفجوة','index':'رقم'}
                            )
                            fig_gap.add_hline(
                                y=info['avg_gap'],
                                line_dash="dash",
                                line_color="red",
                                annotation_text="المتوسط"
                            )
                            st.plotly_chart(
                                fig_gap, use_container_width=True
                            )

            # النافذة الزمنية
            with tab_win:
                st.subheader("تطور الاحتمالات عبر الوقت")
                wins = pat_res['sliding_window']['windows']
                if wins:
                    df_win = pd.DataFrame(wins)
                    fig_win = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=[
                            '% جولات >= 2x عبر الوقت',
                            'متوسط القيمة عبر الوقت'
                        ]
                    )
                    fig_win.add_trace(
                        go.Scatter(
                            x=df_win['window_end'],
                            y=df_win['pct_high'],
                            mode='lines+markers',
                            name='% مرتفع',
                            line=dict(color='green')
                        ), row=1, col=1
                    )
                    fig_win.add_hline(
                        y=float(np.mean(arr>=2)*100),
                        line_dash="dash", line_color="red",
                        row=1, col=1
                    )
                    fig_win.add_trace(
                        go.Scatter(
                            x=df_win['window_end'],
                            y=df_win['avg_value'],
                            mode='lines+markers',
                            name='متوسط القيمة',
                            line=dict(color='blue')
                        ), row=2, col=1
                    )
                    fig_win.update_layout(height=520)
                    st.plotly_chart(fig_win, use_container_width=True)

            # NIST
            with tab_nist:
                st.subheader("🧪 نتائج اختبارات NIST")
                passed_n = int(nist_res['passed_tests'])
                total_n  = int(nist_res['total_tests'])
                verdict  = nist_res['verdict']

                ca, cb, cc = st.columns(3)
                ca.metric("نجح", f"{passed_n}/{total_n}")
                cb.metric("نسبة",f"{int(passed_n/total_n*100)}%")
                cc.metric("الحكم", verdict)

                nist_rows = []
                for r in suite.results.values():
                    nist_rows.append({
                        'الاختبار': r['test_name'],
                        'P-Value' : str(r.get('p_value','-')),
                        'النتيجة' : '✅' if r.get('passed') else '❌',
                        'التفسير' : r.get('interpretation','')
                    })
                st.dataframe(
                    pd.DataFrame(nist_rows),
                    use_container_width=True,
                    hide_index=True
                )

            # ── آخر 50 جولة ─────────────────────────────
            st.markdown("---")
            st.subheader("📊 آخر 50 جولة")
            last50 = raw_data[-50:]
            colors50 = [
                CAT_COLORS[categorize(v)] for v in last50
            ]
            fig_last = go.Figure()
            fig_last.add_trace(go.Bar(
                x=list(range(len(last50))),
                y=last50,
                marker_color=colors50,
                name='القيمة'
            ))
            fig_last.add_hline(
                y=2.0, line_dash="dash",
                line_color="blue",
                annotation_text="2x"
            )
            fig_last.update_layout(
                title="آخر 50 جولة",
                xaxis_title="الجولة",
                yaxis_title="المضاعف",
                height=380
            )
            st.plotly_chart(fig_last, use_container_width=True)

            # ── الاستنتاج ────────────────────────────────
            st.markdown("---")
            st.header("📝 الاستنتاج الأكاديمي")

            sig_lags = suite.results.get(
                'autocorrelation',{}
            ).get('significant_lags',[])
            findings = []
            if sig_lags:
                findings.append(
                    f"ارتباط ذاتي دال في Lag {sig_lags}"
                )
            if not suite.results.get('serial',{}).get('passed',True):
                findings.append("أنماط في الأزواج المتتالية")
            if pat_res['fft_cycles']['has_strong_cycle']:
                findings.append("دورة زمنية قوية في FFT")
            strong_mk2 = [
                p for p in pat_res['markov2']['top_patterns']
                if p['probability'] >= 0.65
            ]
            if strong_mk2:
                findings.append(
                    f"{len(strong_mk2)} نمط Markov-2 بثقة >= 65%"
                )

            if findings:
                st.warning(
                    "### ⚠️ أنماط إحصائية مكتشفة\n\n" +
                    "\n".join(f"• {f}" for f in findings) +
                    "\n\n**ملاحظة أكاديمية:** الأنماط موجودة "
                    "لكن قيمتها العملية محدودة بسبب هامش الكازينو."
                )
            else:
                st.success(
                    "### ✅ لا أنماط إحصائية قوية\n\n"
                    "PRNG يستوفي معايير NIST SP 800-22"
                )

            # ── تحميل التقرير ───────────────────────────
            st.markdown("---")
            report = to_python({
                'summary': {
                    'total_samples': n,
                    'nist_passed'  : nist_res['passed_tests'],
                    'nist_total'   : nist_res['total_tests'],
                    'verdict'      : nist_res['verdict']
                },
                'prediction': {
                    'category'    : pred['final_category'],
                    'label'       : pred['final_label'],
                    'est_value'   : pred['estimated_value'],
                    'value_range' : pred['value_range'],
                    'prob_high'   : pred['prob_high'],
                    'confidence'  : pred['confidence'],
                    'low_streak'  : pred['low_streak']
                },
                'key_findings': findings,
                'top_patterns': {
                    'markov2': pat_res['markov2']['top_patterns'][:5],
                    'markov3': pat_res['markov3']['top_patterns'][:5],
                    'streaks': pat_res['streaks']['streak_results']
                },
                'cycles': {
                    'detected'  : pat_res['fft_cycles']['detected_cycles'],
                    'dominance' : pat_res['fft_cycles']['dominance_ratio']
                },
                'nist_results': {
                    k: {
                        kk: vv
                        for kk, vv in v.items()
                        if kk != 'autocorrelations'
                    }
                    for k, v in suite.results.items()
                }
            })

            st.download_button(
                label="📥 تحميل التقرير الكامل (JSON)",
                data=json.dumps(
                    report, ensure_ascii=False, indent=2
                ),
                file_name="crash_advanced_report.json",
                mime="application/json"
            )

st.markdown("---")
st.caption(
    "🎓 مشروع تخرج أكاديمي | "
    "تحليل PRNG + Markov Chains + FFT + Ensemble Prediction"
)
