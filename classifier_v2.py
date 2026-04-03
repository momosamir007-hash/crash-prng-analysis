# classifier_v2.py
"""
╔══════════════════════════════════════════════════╗
║ 🧠 CLIP Classifier v2 - Fixed Version ║
║ متوافق مع Python 3.14 + Streamlit Cloud ║
╚══════════════════════════════════════════════════╝
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Dict, Optional
import time
import streamlit as st
from candy_elements import (
    ALL_ELEMENTS,
    GameElement,
    ElementCategory
)

class CandyCrushClassifierV2:
    """مصنف شامل لجميع عناصر كاندي كراش"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
        active_categories: List[ElementCategory] = None
    ):
        print("🔄 تحميل CLIP v2...")

        # ═══ تحديد الجهاز ═══
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # ═══ تحميل النموذج ═══
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"❌ خطأ في تحميل النموذج: {e}")
            raise

        # ═══ تصفية العناصر ═══
        if active_categories:
            self.active_elements = {
                k: v for k, v in ALL_ELEMENTS.items()
                if v.category in active_categories
            }
        else:
            self.active_elements = dict(ALL_ELEMENTS)

        # ═══ تجهيز الأوصاف ═══
        self.descriptions = []
        self.desc_to_id = []
        for elem_id, elem in self.active_elements.items():
            for desc in elem.clip_descriptions:
                self.descriptions.append(desc)
                self.desc_to_id.append(elem_id)

        # ═══ تشفير النصوص ═══
        self._precompute_text_features()

        print(f"✅ جاهز! {len(self.active_elements)} عنصر")
        print(f"   {len(self.descriptions)} وصف نصي")
        print(f"   الجهاز: {self.device}")

    # ══════════════════════════════════════════════
    # تشفير النصوص — النسخة المُصححة
    # ══════════════════════════════════════════════
    def _precompute_text_features(self):
        """تشفير كل النصوص مسبقاً"""
        batch_size = 32
        all_features = []
        total_batches = (len(self.descriptions) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(self.descriptions))
            batch_texts = self.descriptions[start:end]

            try:
                # ═══ الإصلاح 1: استخدام tokenizer مباشرة ═══
                text_inputs = self.processor.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                # نقل للجهاز
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

                with torch.no_grad():
                    text_outputs = self.model.get_text_features(**text_inputs)
                    text_outputs = F.normalize(text_outputs, p=2, dim=-1)

                all_features.append(text_outputs.cpu())
            except Exception as e:
                print(f"⚠️ خطأ في تشفير الدفعة {batch_idx}: {e}")
                dummy = torch.zeros(len(batch_texts), 512)
                all_features.append(dummy)

        # ═══ تجميع كل الدفعات ═══
        self.text_features = torch.cat(all_features, dim=0)
        self.text_features = self.text_features.to(self.device)
        print(f"📝 تم تشفير {self.text_features.shape[0]} وصف")

    # ══════════════════════════════════════════════
    # تصنيف خلية واحدة
    # ══════════════════════════════════════════════
    @torch.no_grad()
    def classify_cell(
        self,
        cell_image,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """تصنيف خلية واحدة"""
        # ═══ تحويل الصورة ═══
        if isinstance(cell_image, np.ndarray):
            if len(cell_image.shape) == 3 and cell_image.shape[2] == 3:
                cell_image = Image.fromarray(cell_image)
            else:
                cell_image = Image.fromarray(cell_image)
        if not isinstance(cell_image, Image.Image):
            cell_image = Image.fromarray(
                np.array(cell_image, dtype=np.uint8)
            )
        cell_image = cell_image.convert("RGB")

        try:
            image_inputs = self.processor.image_processor(
                images=cell_image,
                return_tensors="pt"
            )
            pixel_values = image_inputs['pixel_values'].to(self.device)
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, p=2, dim=-1)
        except Exception as e:
            print(f"⚠️ خطأ في تصنيف الخلية: {e}")
            return [("empty", 0.5)]

        # ═══ حساب التشابه ═══
        similarities = torch.matmul(image_features, self.text_features.T).squeeze(0)

        # ═══ تجميع حسب العنصر ═══
        element_scores = {}
        element_counts = {}
        sim_values = similarities.cpu().numpy()
        for idx, elem_id in enumerate(self.desc_to_id):
            score = float(sim_values[idx])
            if elem_id not in element_scores:
                element_scores[elem_id] = 0.0
                element_counts[elem_id] = 0
            element_scores[elem_id] += score
            element_counts[elem_id] += 1

        # المتوسط
        avg_scores = {
            k: element_scores[k] / element_counts[k]
            for k in element_scores
        }

        # Softmax
        keys = list(avg_scores.keys())
        scores_array = np.array([avg_scores[k] for k in keys])
        scores_shifted = scores_array - scores_array.max()
        exp_scores = np.exp(scores_shifted * 15.0)
        sum_exp = exp_scores.sum()
        if sum_exp == 0:
            softmax = np.ones_like(exp_scores) / len(exp_scores)
        else:
            softmax = exp_scores / sum_exp

        ranked_indices = np.argsort(softmax)[::-1]
        results = [
            (keys[i], float(softmax[i]))
            for i in ranked_indices[:top_k]
        ]
        return results

    # ══════════════════════════════════════════════
    # تصنيف دفعة كاملة — النسخة المُصححة
    # ══════════════════════════════════════════════
    @torch.no_grad()
    def classify_batch(
        self,
        cell_images: List,
        batch_size: int = 16
    ) -> List[Tuple[str, float]]:
        """تصنيف دفعة كاملة من الخلايا"""
        all_results = []
        total_batches = (len(cell_images) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(cell_images))
            batch_imgs = cell_images[start:end]

            # ═══ تحويل آمن لـ PIL ═══
            pil_batch = []
            for img in batch_imgs:
                try:
                    if isinstance(img, Image.Image):
                        pil_batch.append(img.convert("RGB"))
                    elif isinstance(img, np.ndarray):
                        if len(img.shape) == 2:  # رمادي → RGB
                            img = np.stack([img, img, img], axis=-1)
                        pil_batch.append(
                            Image.fromarray(img.astype(np.uint8)).convert("RGB")
                        )
                    else:
                        pil_batch.append(Image.new("RGB", (72, 72), (0, 0, 0)))
                except Exception:
                    pil_batch.append(Image.new("RGB", (72, 72), (0, 0, 0)))

            try:
                image_inputs = self.processor.image_processor(
                    images=pil_batch,
                    return_tensors="pt"
                )
                pixel_values = image_inputs['pixel_values'].to(self.device)
                image_features = self.model.get_image_features(pixel_values=pixel_values)
                image_features = F.normalize(image_features, p=2, dim=-1)

                similarities = torch.matmul(image_features, self.text_features.T)
                sim_np = similarities.cpu().numpy()

                for row_idx in range(sim_np.shape[0]):
                    sim_row = sim_np[row_idx]
                    element_scores = {}
                    element_counts = {}
                    for desc_idx, elem_id in enumerate(self.desc_to_id):
                        s = float(sim_row[desc_idx])
                        if elem_id not in element_scores:
                            element_scores[elem_id] = 0.0
                            element_counts[elem_id] = 0
                        element_scores[elem_id] += s
                        element_counts[elem_id] += 1

                    avg = {
                        k: element_scores[k] / element_counts[k]
                        for k in element_scores
                    }
                    keys = list(avg.keys())
                    vals = np.array([avg[k] for k in keys])
                    vals_shifted = vals - vals.max()
                    exp_v = np.exp(vals_shifted * 15.0)
                    sum_v = exp_v.sum()
                    if sum_v > 0:
                        sm = exp_v / sum_v
                    else:
                        sm = np.ones_like(exp_v) / len(exp_v)
                    best_idx = int(np.argmax(sm))
                    all_results.append((keys[best_idx], float(sm[best_idx])))
            except Exception as e:
                print(f"⚠️ خطأ في الدفعة {batch_idx}: {e}")
                for _ in batch_imgs:
                    all_results.append(("empty", 0.1))

        return all_results

    # ══════════════════════════════════════════════
    # دوال مساعدة
    # ══════════════════════════════════════════════
    def get_element_info(self, elem_id: str) -> Optional[GameElement]:
        """معلومات عنصر"""
        return ALL_ELEMENTS.get(elem_id)

    def get_element_emoji(self, elem_id: str) -> str:
        """إيموجي عنصر"""
        elem = ALL_ELEMENTS.get(elem_id)
        return elem.emoji if elem else '❓'
