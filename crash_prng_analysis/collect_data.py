# collect_data.py
"""
جمع بيانات Crash يدوياً أو من مصادر مفتوحة
لأغراض التحليل الإحصائي الأكاديمي
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

class CrashDataCollector:
    """
    جامع البيانات - يعمل مع البيانات المُدخلة يدوياً
    أو من ملفات CSV تم تصديرها من اللعبة
    """
    
    def __init__(self):
        self.data = []
        self.metadata = {
            'collected_at': datetime.now().isoformat(),
            'source': 'manual_collection',
            'purpose': 'academic_research'
        }
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """تحميل من ملف CSV"""
        df = pd.read_csv(filepath)
        self.data = df['crash_point'].tolist()
        return df
    
    def load_from_list(self, values: list) -> None:
        """تحميل من قائمة مباشرة"""
        self.data = [float(v) for v in values]
    
    def save_to_csv(self, filepath: str) -> None:
        """حفظ البيانات"""
        df = pd.DataFrame({
            'round': range(1, len(self.data) + 1),
            'crash_point': self.data,
            'is_high': [1 if v >= 2.0 else 0 for v in self.data],
            'log_value': [np.log(max(v, 1.0)) for v in self.data]
        })
        df.to_csv(filepath, index=False)
        print(f"✅ تم حفظ {len(self.data)} سجل في {filepath}")
    
    def get_summary(self) -> dict:
        arr = np.array(self.data)
        return {
            'total_rounds': len(arr),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'pct_above_2x': float(np.mean(arr >= 2.0)),
            'pct_above_5x': float(np.mean(arr >= 5.0)),
            'pct_above_10x': float(np.mean(arr >= 10.0))
        }
