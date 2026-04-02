# move_engine.py
"""
╔════════════════════════════════════════════════════════╗
║ 🧠 Candy Crush Move Engine v2.0 ║
║ محرك بحث متقدم مع كشف السلاسل والحلوى الخاصة ║
╚════════════════════════════════════════════════════════╝
"""
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field

@dataclass
class MatchResult:
    """نتيجة تطابق"""
    cells: Set[Tuple[int, int]]  # الخلايا المتطابقة
    color: str                   # لون الحلوى
    length: int                  # طول التطابق
    direction: str               # أفقي أو عمودي
    special_candy: str = "none"  # نوع الحلوى الخاصة الناتجة

@dataclass
class MoveResult:
    """نتيجة حركة كاملة"""
    pos1: Tuple[int, int]
    pos2: Tuple[int, int]
    direction: str
    score: int
    matches: List[MatchResult]
    chain_depth: int = 1
    special_candies: List[str] = field(default_factory=list)
    details: str = ""
    priority: float = 0.0

class CandyEngine:
    """
    ╔══════════════════════════════════════════╗
    ║ محرك الحركات الذكي ║
    ║ ✓ كشف التطابقات (3, 4, 5+) ║
    ║ ✓ كشف الحلوى الخاصة ║
    ║ ✓ محاكاة السلاسل (Cascades) ║
    ║ ✓ تقييم استراتيجي متعدد العوامل ║
    ╚══════════════════════════════════════════╝
    """

    # نظام النقاط
    SCORES = {
        3: 10,          # تطابق عادي
        4: 30,          # حلوى مخططة (Striped)
        5: 100,         # قنبلة ألوان (Color Bomb)
        'wrapped': 50,  # حلوى مغلفة (تقاطع)
        'cascade_bonus': 15,  # مكافأة لكل مرحلة سلسلة
    }
    BLOCKED = {'empty', 'blocker', 'unknown', 'chocolate', 'ice'}

    def __init__(self, grid_matrix: np.ndarray):
        self.grid = grid_matrix.copy()
        self.rows, self.cols = self.grid.shape

    # ═══════════════════════════════════════
    # البحث الرئيسي عن الحركات
    # ═══════════════════════════════════════
    def find_all_moves(self) -> List[Dict]:
        """البحث الشامل عن جميع الحركات الصالحة مع تقييم متقدم"""
        valid_moves = []

        for r in range(self.rows):
            for c in range(self.cols):
                current = self.grid[r, c]
                if current in self.BLOCKED:
                    continue

                # ═══ تبديل يميناً ═══
                if c < self.cols - 1:
                    neighbor = self.grid[r, c + 1]
                    if (neighbor not in self.BLOCKED and current != neighbor):
                        result = self._evaluate_move(
                            (r, c), (r, c + 1), 'Right ➡️'
                        )
                        if result and result['score'] > 0:
                            valid_moves.append(result)

                # ═══ تبديل لأسفل ═══
                if r < self.rows - 1:
                    neighbor = self.grid[r + 1, c]
                    if (neighbor not in self.BLOCKED and current != neighbor):
                        result = self._evaluate_move(
                            (r, c), (r + 1, c), 'Down ⬇️'
                        )
                        if result and result['score'] > 0:
                            valid_moves.append(result)

        # ═══ الترتيب المتقدم (متعدد العوامل) ═══
        for move in valid_moves:
            move['priority'] = self._calculate_priority(move)

        valid_moves.sort(key=lambda x: x['priority'], reverse=True)
        return valid_moves

    # ═══════════════════════════════════════
    # تقييم حركة واحدة
    # ═══════════════════════════════════════
    def _evaluate_move(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
        direction: str
    ) -> Dict:
        """تقييم شامل لحركة واحدة"""
        # نسخة مؤقتة
        temp = np.copy(self.grid)
        temp[pos1], temp[pos2] = temp[pos2], temp[pos1]

        # البحث عن تطابقات
        all_matches = self._find_all_matches(temp)
        if not all_matches:
            return None

        # حساب النقاط
        total_score = 0
        details_parts = []
        specials = []

        for match in all_matches:
            points = self.SCORES.get(match.length, 0)
            if match.length > 5:
                points = self.SCORES[5]
            total_score += points

            # كشف الحلوى الخاصة
            special = self._detect_special(match, all_matches)
            if special != "none":
                specials.append(special)
                if special == "wrapped":
                    total_score += self.SCORES['wrapped']

            detail = f"{'H' if match.direction == 'horizontal' else 'V'}"
            detail += f"-{match.length}"
            if special != "none":
                detail += f" ({special})"
            details_parts.append(detail)

        # ═══ محاكاة السلسلة (Cascade) ═══
        chain_depth, chain_score = self._simulate_cascade(temp, all_matches)
        total_score += chain_score
        if chain_depth > 1:
            details_parts.append(f"Chain x{chain_depth}!")

        return {
            'pos1': pos1,
            'pos2': pos2,
            'direction': direction,
            'score': total_score,
            'matches': all_matches,
            'chain_depth': chain_depth,
            'special_candies': specials,
            'details': " | ".join(details_parts),
            'matched_count': sum(len(m.cells) for m in all_matches),
            'candy1': self.grid[pos1],
            'candy2': self.grid[pos2],
        }

    # ═══════════════════════════════════════
    # البحث عن التطابقات في الشبكة
    # ═══════════════════════════════════════
    def _find_all_matches(self, grid: np.ndarray) -> List[MatchResult]:
        """إيجاد جميع التطابقات (3+) في الشبكة"""
        matches = []
        visited_h = set()
        visited_v = set()
        rows, cols = grid.shape

        # ═══ بحث أفقي ═══
        for r in range(rows):
            c = 0
            while c < cols:
                color = grid[r, c]
                if color in self.BLOCKED:
                    c += 1
                    continue
                # عد المتتالية
                run_start = c
                while c < cols and grid[r, c] == color:
                    c += 1
                run_len = c - run_start
                if run_len >= 3:
                    cells = {(r, ci) for ci in range(run_start, c)}
                    if frozenset(cells) not in visited_h:
                        visited_h.add(frozenset(cells))
                        matches.append(MatchResult(
                            cells=cells,
                            color=color,
                            length=run_len,
                            direction='horizontal'
                        ))

        # ═══ بحث عمودي ═══
        for c in range(cols):
            r = 0
            while r < rows:
                color = grid[r, c]
                if color in self.BLOCKED:
                    r += 1
                    continue
                run_start = r
                while r < rows and grid[r, c] == color:
                    r += 1
                run_len = r - run_start
                if run_len >= 3:
                    cells = {(ri, c) for ri in range(run_start, r)}
                    if frozenset(cells) not in visited_v:
                        visited_v.add(frozenset(cells))
                        matches.append(MatchResult(
                            cells=cells,
                            color=color,
                            length=run_len,
                            direction='vertical'
                        ))

        return matches

    # ═══════════════════════════════════════
    # كشف الحلوى الخاصة
    # ═══════════════════════════════════════
    def _detect_special(
        self,
        match: MatchResult,
        all_matches: List[MatchResult]
    ) -> str:
        """تحديد نوع الحلوى الخاصة الناتجة"""
        # قنبلة ألوان (5 في صف)
        if match.length >= 5:
            match.special_candy = "color_bomb"
            return "color_bomb 💣"

        # حلوى مخططة (4 في صف)
        if match.length == 4:
            match.special_candy = "striped"
            return "striped 🎯"

        # حلوى مغلفة (تقاطع L أو T)
        for other in all_matches:
            if other is match:
                continue
            if other.color == match.color:
                # هل يتقاطعان؟
                intersection = match.cells & other.cells
                if intersection:
                    match.special_candy = "wrapped"
                    other.special_candy = "wrapped"
                    return "wrapped 🎁"

        match.special_candy = "none"
        return "none"

    # ═══════════════════════════════════════
    # محاكاة السلاسل (Cascades)
    # ═══════════════════════════════════════
    def _simulate_cascade(
        self,
        grid: np.ndarray,
        initial_matches: List[MatchResult],
        max_depth: int = 5
    ) -> Tuple[int, int]:
        """محاكاة تأثير الجاذبية بعد إزالة التطابقات"""
        temp = np.copy(grid)
        total_cascade_score = 0
        depth = 1

        # إزالة التطابقات الأولى
        for match in initial_matches:
            for (r, c) in match.cells:
                temp[r, c] = 'empty'

        while depth < max_depth:
            # تطبيق الجاذبية
            temp = self._apply_gravity(temp)

            # البحث عن تطابقات جديدة
            new_matches = self._find_all_matches(temp)
            if not new_matches:
                break

            depth += 1

            # حساب نقاط السلسلة
            for match in new_matches:
                points = self.SCORES.get(match.length, 0)
                cascade_bonus = self.SCORES['cascade_bonus'] * depth
                total_cascade_score += points + cascade_bonus

                # إزالة
                for (r, c) in match.cells:
                    temp[r, c] = 'empty'

        return depth, total_cascade_score

    def _apply_gravity(self, grid: np.ndarray) -> np.ndarray:
        """إسقاط الحلوى لملء الفراغات"""
        temp = np.copy(grid)
        rows, cols = temp.shape

        for c in range(cols):
            # تجميع غير الفارغة من الأسفل
            column = []
            for r in range(rows - 1, -1, -1):
                if temp[r, c] != 'empty':
                    column.append(temp[r, c])

            # إعادة الملء من الأسفل
            for r in range(rows - 1, -1, -1):
                if column:
                    temp[r, c] = column.pop(0)
                else:
                    temp[r, c] = 'empty'

        return temp

    # ═══════════════════════════════════════
    # حساب الأولوية المركبة
    # ═══════════════════════════════════════
    def _calculate_priority(self, move: Dict) -> float:
        """
        تقييم استراتيجي يجمع عدة عوامل:
        - النقاط المباشرة
        - عمق السلسلة
        - الحلوى الخاصة
        - عدد الحلوى المحذوفة
        - الموقع (الأسفل أفضل أحياناً)
        """
        score = move['score']
        chain = move.get('chain_depth', 1)
        specials = len(move.get('special_candies', []))
        matched = move.get('matched_count', 3)
        row = move['pos1'][0]

        priority = (
            score * 1.0 +
            chain * 20 +
            specials * 40 +
            matched * 5 +
            (row / self.rows) * 8  # تفضيل الصفوف السفلية قليلاً
        )
        return round(priority, 1)

    # ═══════════════════════════════════════
    # دوال المساعدة
    # ═══════════════════════════════════════
    def get_board_stats(self) -> Dict:
        """إحصائيات اللوحة"""
        stats = {}
        for r in range(self.rows):
            for c in range(self.cols):
                candy = self.grid[r, c]
                stats[candy] = stats.get(candy, 0) + 1
        return stats

    def get_grid_text(self) -> str:
        """تمثيل نصي للشبكة"""
        emoji = {
            'red': '🔴',
            'blue': '🔵',
            'green': '🟢',
            'yellow': '🟡',
            'orange': '🟠',
            'purple': '🟣',
            'empty': '⬜'
        }
        lines = []
        for r in range(self.rows):
            row_emojis = []
            for c in range(self.cols):
                row_emojis.append(emoji.get(self.grid[r, c], '❓'))
            lines.append(' '.join(row_emojis))
        return '\n'.join(lines)

    def print_best_moves(self, top_n: int = 3):
        """طباعة أفضل الحركات"""
        moves = self.find_all_moves()
        if not moves:
            print("\n❌ لا توجد حركات متاحة!")
            return

        medals = ['🥇', '🥈', '🥉'] + ['🎯'] * 10
        print(f"\n{'═' * 55}")
        print(f" 🎮 أفضل {min(top_n, len(moves))} حركات")
        print(f"{'═' * 55}")

        for i, move in enumerate(moves[:top_n]):
            p1, p2 = move['pos1'], move['pos2']
            print(f"\n{medals[i]} الخيار {i + 1}:")
            print(f" بدّل [{move['candy1']}] في {p1}")
            print(f" {move['direction']} مع [{move['candy2']}] في {p2}")
            print(f" 📊 النقاط: {move['score']}")
            print(f" ⛓️ السلسلة: x{move['chain_depth']}")
            if move['special_candies']:
                print(f" ⭐ خاصة: {', '.join(move['special_candies'])}")
            print(f" 📝 {move['details']}")
            print(f" 🏆 الأولوية: {move['priority']}")
        print(f"{'─' * 55}")

# ═══ اختبار سريع ═══
if __name__ == "__main__":
    test = np.array([
        ['red', 'blue', 'green', 'yellow', 'red', 'orange', 'purple', 'blue', 'green'],
        ['yellow', 'red', 'red', 'blue', 'green', 'yellow', 'red', 'orange', 'purple'],
        ['blue', 'green', 'yellow', 'red', 'orange', 'purple', 'blue', 'green', 'yellow'],
        ['red', 'orange', 'blue', 'green', 'yellow', 'red', 'orange', 'purple', 'blue'],
        ['green', 'yellow', 'red', 'red', 'red', 'purple', 'blue', 'green', 'yellow'],
        ['red', 'blue', 'green', 'yellow', 'orange', 'red', 'orange', 'purple', 'blue'],
        ['yellow', 'orange', 'purple', 'blue', 'green', 'yellow', 'red', 'blue', 'green'],
        ['blue', 'green', 'yellow', 'red', 'orange', 'purple', 'blue', 'green', 'red'],
        ['red', 'blue', 'green', 'yellow', 'red', 'orange', 'purple', 'red', 'red'],
    ])
    engine = CandyEngine(test)
    engine.print_best_moves(5)
