import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import math
import os

st.set_page_config(page_title="Candy Crush AI", layout="centered")
st.title("🍬 العرّاف: مساعد كاندي كراش")
st.markdown("---")

from ultralytics import YOLO
from move_engine import CandyEngine

# ─── دالة رسم سهم باستخدام Pillow فقط ───
def draw_arrow(draw, start, end, color=(0, 255, 0), width=8):
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    arrow_len = 30
    arrow_angle = math.pi / 6
    x1 = end[0] - arrow_len * math.cos(angle - arrow_angle)
    y1 = end[1] - arrow_len * math.sin(angle - arrow_angle)
    x2 = end[0] - arrow_len * math.cos(angle + arrow_angle)
    y2 = end[1] - arrow_len * math.sin(angle + arrow_angle)
    draw.polygon([end, (int(x1), int(y1)), (int(x2), int(y2))], fill=color)

# ─── دالة رسم دائرة شفافة ───
def draw_highlight(image, center, radius, color=(0, 255, 0, 100)):
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    x, y = center
    overlay_draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color
    )
    image_rgba = image.convert('RGBA')
    result = Image.alpha_composite(image_rgba, overlay)
    return result.convert('RGB')

# ─── تحميل النموذج ───
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

model = load_model()
if model is None:
    st.error("❌ ملف best.pt غير موجود!")
    st.stop()

st.sidebar.success("✅ النموذج جاهز للعمل!")

class_map = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'yellow',
    4: 'orange',
    5: 'purple',
    6: 'blocker'
}

st.info("💡 ارفع صورة مقصوصة تحتوي على لوحة اللعب فقط (9x9)")

uploaded_file = st.file_uploader("📷 ارفع صورة اللوحة...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if st.button("🚀 تحليل ورسم أفضل حركة"):
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        h, w, _ = img_np.shape

        with st.spinner("الذكاء الاصطناعي يقرأ اللوحة..."):
            results = model.predict(img_np, conf=0.35)
            boxes = results[0].boxes
            grid = np.full((9, 9), 'empty', dtype=object)
            cell_w = w / 9.0
            cell_h = h / 9.0

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].item())
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                col = int(cx // cell_w)
                row = int(cy // cell_h)
                col = min(max(col, 0), 8)
                row = min(max(row, 0), 8)
                grid[row, col] = class_map.get(cls, 'empty')

            engine = CandyEngine(grid)
            moves = engine.find_all_moves()

        st.markdown("---")
        st.subheader("💡 الحل البصري المقترح:")

        if moves:
            best_move = moves[0]
            r1, c1 = best_move['pos1']
            r2, c2 = best_move['pos2']

            p1x = int((c1 + 0.5) * cell_w)
            p1y = int((r1 + 0.5) * cell_h)
            p2x = int((c2 + 0.5) * cell_w)
            p2y = int((r2 + 0.5) * cell_h)

            p1_pixel = (p1x, p1y)
            p2_pixel = (p2x, p2y)

            # رسم الدائرة الشفافة
            radius = int(min(cell_w, cell_h) * 0.4)
            result_img = draw_highlight(image, p1_pixel, radius)

            # رسم السهم
            draw = ImageDraw.Draw(result_img)
            draw_arrow(draw, p1_pixel, p2_pixel, color=(0, 255, 0), width=8)

            st.image(result_img, caption="🥇 أفضل حركة!", use_container_width=True)

            st.success(
                "حرك من الصف " + str(r1) + " العمود " + str(c1) + " ➜ " + best_move['direction']
            )
            st.info(
                "✨ النتيجة: " + str(best_move['score']) + " نقطة (" + best_move['details'] + ")"
            )
        else:
            st.image(image, use_container_width=True)
            st.warning("⚠️ لم يتم العثور على حركات.")
