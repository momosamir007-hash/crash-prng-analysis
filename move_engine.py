# move_engine.py
"""
╔════════════════════════════════════════════════════════╗
║ 🧠 Candy Crush Move Engine                             ║
║ محرك البحث الاستراتيجي لاكتشاف أفضل الحركات الممكنة    ║
╚════════════════════════════════════════════════════════╝
"""
import numpy as np
from typing import List, Dict, Tuple
import copy

class CandyEngine:
    def __init__(self, grid_matrix: np.ndarray):
        """
        يستقبل مصفوفة ثنائية الأبعاد (Numpy Array) تحتوي على أسماء الحلوى كـ Strings.
        مثال: 'red', 'blue', 'empty'
        """
        self.grid = grid_matrix
        self.rows, self.cols = self.grid.shape

    def find_all_moves(self) -> List[Dict]:
        """
        البحث عن جميع الحركات الصالحة في الشبكة وتقييمها.
        Returns:
            قائمة بالقواميس، كل قاموس يحتوي على تفاصيل الحركة ودرجتها.
        """
        valid_moves = []

        for r in range(self.rows):
            for c in range(self.cols):
                current_candy = self.grid[r, c]
                if current_candy in ['empty', 'blocker', 'unknown']:
                    continue

                # 1. تجربة التبديل يميناً (Horizontal Swap)
                if c < self.cols - 1:
                    right_candy = self.grid[r, c + 1]
                    if right_candy not in ['empty', 'blocker', 'unknown'] and current_candy != right_candy:
                        score, match_details = self._evaluate_swap((r, c), (r, c + 1))
                        if score > 0:
                            valid_moves.append({
                                'pos1': (r, c),
                                'pos2': (r, c + 1),
                                'direction': 'Right ➡️',
                                'score': score,
                                'details': match_details
                            })

                # 2. تجربة التبديل لأسفل (Vertical Swap)
                if r < self.rows - 1:
                    down_candy = self.grid[r + 1, c]
                    if down_candy not in ['empty', 'blocker', 'unknown'] and current_candy != down_candy:
                        score, match_details = self._evaluate_swap((r, c), (r + 1, c))
                        if score > 0:
                            valid_moves.append({
                                'pos1': (r, c),
                                'pos2': (r + 1, c),
                                'direction': 'Down ⬇️',
                                'score': score,
                                'details': match_details
                            })

        # ترتيب الحركات من الأعلى تقييماً إلى الأقل
        valid_moves.sort(key=lambda x: x['score'], reverse=True)
        return valid_moves

    def _evaluate_swap(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> Tuple[int, str]:
        """
        محاكاة التبديل وحساب النتيجة.
        """
        # إنشاء نسخة وهمية من الشبكة للتجربة
        temp_grid = np.copy(self.grid)
        
        # تنفيذ التبديل
        temp_grid[pos1[0], pos1[1]], temp_grid[pos2[0], pos2[1]] = temp_grid[pos2[0], pos2[1]], temp_grid[pos1[0], pos1[1]]

        # فحص التطابقات الناتجة عن الحركة في كلا الموقعين
        matches1 = self._check_matches_at(temp_grid, pos1[0], pos1[1])
        matches2 = self._check_matches_at(temp_grid, pos2[0], pos2[1])

        total_score = 0
        details = []

        # تقييم التطابقات
        for m in [matches1, matches2]:
            if m['h_len'] >= 3:
                total_score += self._calculate_score(m['h_len'])
                details.append(f"Horizontal Match {m['h_len']}")
            if m['v_len'] >= 3:
                total_score += self._calculate_score(m['v_len'])
                details.append(f"Vertical Match {m['v_len']}")
            
            # تقاطع يعطي حلوى مغلفة (Wrapped Candy)
            if m['h_len'] >= 3 and m['v_len'] >= 3:
                total_score += 50 # نقاط إضافية لصناعة حلوى مغلفة
                details.append("Wrapped Candy Created! 🎁")

        return total_score, " | ".join(details)

    def _check_matches_at(self, grid: np.ndarray, r: int, c: int) -> Dict:
        """
        البحث عن طول التطابق الأفقي والعمودي لنقطة معينة.
        """
        color = grid[r, c]
        if color == 'empty':
            return {'h_len': 0, 'v_len': 0}

        # فحص أفقي
        left = c
        while left > 0 and grid[r, left - 1] == color:
            left -= 1
        right = c
        while right < self.cols - 1 and grid[r, right + 1] == color:
            right += 1
        h_len = right - left + 1

        # فحص عمودي
        up = r
        while up > 0 and grid[up - 1, c] == color:
            up -= 1
        down = r
        while down < self.rows - 1 and grid[down + 1, c] == color:
            down += 1
        v_len = down - up + 1

        return {'h_len': h_len, 'v_len': v_len}

    def _calculate_score(self, match_length: int) -> int:
        """
        نظام تنقيط يحاكي منطق اللعبة:
        3 = تطابق عادي
        4 = حلوى مخططة (أهمية عالية)
        5 = قنبلة ألوان (الأهمية القصوى)
        """
        if match_length == 3:
            return 10
        elif match_length == 4:
            return 30 # Striped
        elif match_length >= 5:
            return 100 # Color Bomb
        return 0

    def print_best_moves(self, top_n: int = 3):
        """طباعة أفضل الحركات بشكل أنيق"""
        moves = self.find_all_moves()
        if not moves:
            print("\n❌ لا توجد حركات متاحة على هذه اللوحة!")
            return

        print(f"\n🎯 أفضل {min(top_n, len(moves))} حركات مقترحة:")
        print("═" * 50)
        for i, move in enumerate(moves[:top_n]):
            p1, p2 = move['pos1'], move['pos2']
            candy1, candy2 = self.grid[p1], self.grid[p2]
            print(f"🥇 الخيار {i+1}:") if i == 0 else print(f"🥈 الخيار {i+1}:")
            print(f"   بدّل [{candy1}] في الموقع {p1}")
            print(f"   {move['direction']} مع [{candy2}] في الموقع {p2}")
            print(f"   النتيجة المتوقعة: {move['score']} نقطة ({move['details']})")
            print("─" * 50)

# ---------------------------------------------------------
# كيفية استخدامه مع كودك الأساسي (في ملف main.py):
# ---------------------------------------------------------
if __name__ == "__main__":
    # مصفوفة تجريبية سريعة لاختبار المحرك بمعزل عن الرؤية الحاسوبية
    test_grid = np.array([
        ['red', 'blue', 'green', 'yellow'],
        ['red', 'orange', 'purple', 'green'],
        ['blue', 'red', 'red', 'yellow'],
        ['yellow', 'orange', 'blue', 'blue']
    ])
    
    engine = CandyEngine(test_grid)
    engine.print_best_moves()
