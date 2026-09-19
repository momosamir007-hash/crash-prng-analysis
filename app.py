# -*- coding: utf-8 -*-
"""
Reconstruction of excel_inspector_pro.py from a Python 3.14 PyInstaller bytecode object.

IMPORTANT:
    This is a readable reconstruction of the recovered bytecode, not a byte-for-byte
    recovery of the original source file. Names, strings, constants and the main
    control-flow/behaviour were recovered from the embedded code object; formatting,
    comments and some compiler-level details are necessarily reconstructed.
"""

import sys
import os
import json
import subprocess
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QFileDialog, QComboBox,
    QLabel, QMessageBox, QHeaderView, QDialog, QDoubleSpinBox, QTextEdit,
    QTabWidget, QListWidget, QListWidgetItem, QSplitter,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QIcon, QDragEnterEvent, QDropEvent

import openpyxl
from openpyxl.styles import PatternFill


def resource_path(relative_path):
    """Return a resource path compatible with PyInstaller and normal Python."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def detect_subject_type(text):
    """Detect the subject family from a sheet/file label."""
    t = str(text).lower()
    if any(k.lower() in t for k in ("فرنسية", "Français", "Fr")):
        return "الفرنسية"
    if any(k.lower() in t for k in ("انجليزية", "Anglais", "Ang")):
        return "الانجليزية"
    if any(k.lower() in t for k in ("بدنية", "رياضة", "Sport", "EPS")):
        return "الرياضة"
    return "العربية"


def open_file(path):
    """Open a file using the platform's default application."""
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", path])


def get_mark_cols(df, header_row):
    """Return columns considered grade/mark columns, excluding identity/Obs columns."""
    excluded = frozenset({"", "nan", "nom", "date_n", "matricule", "obs", "prenom"})
    mark_cols = []
    for c in range(df.shape[1]):
        h = str(df.iloc[header_row, c]).strip().lower()
        if h not in excluded:
            mark_cols.append(c)
    return mark_cols


def get_obs_col(df, header_row):
    """Find the Obs column index in a sheet."""
    for c in range(df.shape[1]):
        h = str(df.iloc[header_row, c]).strip().lower()
        if h == "obs":
            return c
    return None


class ObservationSettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("إعدادات الملاحظات والمجالات")
        self.resize(860, 620)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.settings = current_settings or self.get_default_settings()
        self.tabs = {}
        self.init_ui()

    def get_default_settings(self):
        ar = [
            (0, 1.99, "ضعيف جدا"),
            (2, 3.99, "ضعيف يجب العمل أكثر"),
            (4, 4.99, "دون الوسط"),
            (5, 5.99, "فوق المتوسط"),
            (6, 7.5, "قريب من الجيد"),
            (7.51, 8.99, "جيد"),
            (9, 10, "ممتاز"),
        ]
        r = {
            "العربية": list(ar),
            "الفرنسية": [
                (0, 1.99, "Très faible"), (2, 3.99, "Faible"),
                (4, 4.99, "Insuffisant"), (5, 5.99, "Passable"),
                (6, 7.5, "Assez bien"), (7.51, 8.99, "Bien"),
                (9, 10, "Excellent"),
            ],
            "الانجليزية": [
                (0, 1.99, "Very weak"), (2, 3.99, "Weak"),
                (4, 4.99, "Below average"), (5, 5.99, "Above average"),
                (6, 7.5, "Fairly good"), (7.51, 8.99, "Good"),
            ],
            "الرياضة": list(ar),
        }
        return r

    def init_ui(self):
        layout = QVBoxLayout(self)
        tab_widget = QTabWidget()
        tabs = ("العربية", "الفرنسية", "الانجليزية", "الرياضة")

        for sub in tabs:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tbl = QTableWidget()
            tbl.setHorizontalHeaderLabels(("من", "إلى", "الملاحظة"))
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            tbl.setFont(QFont("Arial"))
            data = self.settings.get(sub, [])
            tbl.setRowCount(len(data))
            for r, rule in enumerate(data):
                sp_min = QDoubleSpinBox()
                sp_min.setRange(0, 10)
                sp_min.setSingleStep(0.01)
                sp_min.setValue(float(rule[0]))
                sp_max = QDoubleSpinBox()
                sp_max.setRange(0, 10)
                sp_max.setSingleStep(0.01)
                sp_max.setValue(float(rule[1]))
                tbl.setCellWidget(r, 0, sp_min)
                tbl.setCellWidget(r, 1, sp_max)
                tbl.setItem(r, 2, QTableWidgetItem(str(rule[2])))
            tab_layout.addWidget(tbl)
            tabs and self.tabs.__setitem__(sub, tbl)
            tab_widget.addTab(tab, sub)

        layout.addWidget(tab_widget)
        btn_row = QHBoxLayout()
        b_exp = QPushButton("📤 تصدير القواعد (TXT)")
        b_exp.setStyleSheet("background-color:#2d3436;color:white;padding:8px;")
        b_exp.clicked.connect(self.export_settings)
        b_imp = QPushButton("📥 استيراد القواعد (TXT)")
        b_imp.setStyleSheet("background-color:#2d3436;color:white;padding:8px;")
        b_imp.clicked.connect(self.import_settings)
        b_ok = QPushButton("✅ حفظ الإعدادات")
        b_ok.setStyleSheet("background-color:#27ae60;color:white;font-weight:bold;padding:8px;")
        b_ok.clicked.connect(self.accept)
        btn_row.addWidget(b_exp)
        btn_row.addWidget(b_imp)
        btn_row.addStretch()
        btn_row.addWidget(b_ok)
        layout.addLayout(btn_row)

    def get_updated_settings(self):
        result = {}
        for sub, tbl in self.tabs.items():
            rows = []
            for r in range(tbl.rowCount()):
                mi = tbl.cellWidget(r, 0).value()
                ma = tbl.cellWidget(r, 1).value()
                item = tbl.item(r, 2)
                tx = item.text() if item else ""
                rows.append((mi, ma, tx))
            result[sub] = rows
        return result

    def export_settings(self):
        p, _ = QFileDialog.getSaveFileName(self, "تصدير", "", "TXT (*.txt)")
        if not p:
            return
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.get_updated_settings(), f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "تم", "تم تصدير القواعد.")

    def import_settings(self):
        p, _ = QFileDialog.getOpenFileName(self, "استيراد", "", "TXT (*.txt)")
        if not p:
            return
        with open(p, "r", encoding="utf-8") as f:
            imp = json.load(f)
        for s, d in imp.items():
            if s not in self.tabs:
                continue
            tbl = self.tabs[s]
            tbl.setRowCount(len(d))
            for r, rule in enumerate(d):
                sp_min = tbl.cellWidget(r, 0)
                sp_max = tbl.cellWidget(r, 1)
                if sp_min is None:
                    sp_min = QDoubleSpinBox(); sp_min.setRange(0, 10); sp_min.setSingleStep(0.01)
                    tbl.setCellWidget(r, 0, sp_min)
                if sp_max is None:
                    sp_max = QDoubleSpinBox(); sp_max.setRange(0, 10); sp_max.setSingleStep(0.01)
                    tbl.setCellWidget(r, 1, sp_max)
                sp_min.setValue(float(rule[0]))
                sp_max.setValue(float(rule[1]))
                tbl.setItem(r, 2, QTableWidgetItem(str(rule[2])))
        QMessageBox.information(self, "تم", "تم استيراد القواعد.")


class BatchErrorReportDialog(QDialog):
    def __init__(self, parent, html_content, raw_text_for_export):
        super().__init__(parent)
        self.setWindowTitle("تقرير فحص الدفعة (أخطاء وحالة الملفات)")
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.resize(800, 600)
        self.raw_text_for_export = raw_text_for_export
        vl = QVBoxLayout(self)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setHtml(html_content)
        vl.addWidget(te)
        btn_row = QHBoxLayout()
        b_export = QPushButton("📄 تصدير التقرير المجمع (TXT)")
        b_export.setStyleSheet("background-color:#2d3436; color:white; padding:10px; font-weight:bold;")
        b_export.clicked.connect(self.export_report)
        b_open = QPushButton("✏️  فهمت – أغلق التقرير")
        b_open.setStyleSheet("background-color:#e74c3c; color:white; padding:10px; font-weight:bold;")
        b_open.clicked.connect(self.accept)
        btn_row.addWidget(b_export)
        btn_row.addStretch()
        btn_row.addWidget(b_open)
        vl.addLayout(btn_row)

    def export_report(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "حفظ التقرير المجمع", "تقرير_الدفعة.txt", "Text Files (*.txt)"
        )
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.raw_text_for_export)
            QMessageBox.information(self, "نجاح", "تم تصدير التقرير بنجاح.")
        except Exception as e:
            QMessageBox.warning(self, "خطأ", "تعذر حفظ الملف: " + str(e))


class ExcelInspectorApp(QMainWindow):
    HEADER_ROW = 4
    DATA_ROW = 5

    def __init__(self):
        super().__init__()
        self.setWindowTitle("برنامج فحص وتصحيح نقاط الرقمنة الإحترافي (نسخة الدفعات) ✔")
        self.resize(1400, 900)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.setAcceptDrops(True)

        self.loaded_files = []
        self.current_file_path = None
        self.excel_data = {}
        self.current_sheet = None
        self.current_subject_type = "العربية"

        self.color_empty = QColor("#ff7675")
        self.color_comma = QColor("#ffeaa7")
        self.color_range = QColor("#fab1a0")
        self._loading = False

        icon_path = resource_path(os.path.join("img", "001.png"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.obs_settings = ObservationSettingsDialog().get_default_settings()
        self.init_ui()

    def dragEnterEvent(self, event: QDragEnterEvent):
        urls = event.mimeData().urls()
        valid = False
        for u in urls:
            local_f = u.toLocalFile()
            if os.path.isdir(local_f) or local_f.lower().endswith((".xlsx", ".xls")):
                valid = True
                break
        if valid:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths = []
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if os.path.isdir(local_path):
                for f in os.listdir(local_path):
                    if f.lower().endswith((".xlsx", ".xls")) and not f.startswith("~$"):
                        paths.append(os.path.join(local_path, f))
            elif local_path.lower().endswith((".xlsx", ".xls")):
                paths.append(local_path)
        if paths:
            self.add_files_to_list(paths)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)
        central.setObjectName("central")
        central.setStyleSheet("""
            QMainWindow, QWidget#central { background-color: #f8f9fa; }
            QPushButton { font-size:14px; padding:10px 18px; border-radius:8px; font-weight:bold;
                          border:none; color:white; }
            QPushButton:hover { background-color: rgba(0,0,0,0.1); }
            QPushButton:disabled { background-color:#dcdde1; color:#7f8fa6; }
            QLabel { font-size:14px; font-weight:bold; color:#2c3e50; }
            QComboBox { font-size:14px; padding:8px 12px; border:2px solid #3498db;
                        border-radius:6px; background-color:white; color:#2c3e50; min-width:220px; }
            QComboBox::drop-down { border:none; }
            QComboBox QAbstractItemView { background-color:white; selection-background-color:#ecf0f1; color:#2c3e50; }
            QTableWidget { font-size:14px; gridline-color:#e1e8ed; background-color:white;
                           border:1px solid #ced4da; border-radius:6px; alternate-background-color:#fbfbfc; }
            QTableWidget::item:selected { background-color:#3498db; color:white; }
            QHeaderView::section { background-color:#2c3e50; color:white; font-weight:bold;
                                   padding:8px; border:1px solid #34495e; }
            QListWidget { font-size:13px; background-color:white; border:1px solid #ced4da;
                          border-radius:6px; padding:5px; }
            QListWidget::item { padding:8px; border-bottom:1px solid #f1f2f6; }
            QListWidget::item:selected { background-color:#3498db; color:white; border-radius:4px; }
            QListWidget::item:hover { background-color:#ecf0f1; }
        """)

        logo_row = QHBoxLayout()
        logo = QLabel()
        logo.setText('<span style="font-family: Arial; font-size: 22px; font-weight: bold;">'
                     '<span style="color:#2980b9;">برنامج </span>'
                     '<span style="color:#c0392b;">الرقمنة </span>'
                     '<span style="color:#27ae60;">الإحترافي</span></span>'
                     '<span style="color:#7f8fa6; font-size:14px;"> '
                     '(يدعم معالجة الدفعات - اسحب مجلد أو ملفات هنا)</span>')
        logo_row.addWidget(logo)
        logo_row.addStretch()
        root.addLayout(logo_row)

        bar = QHBoxLayout()
        btn_add_files = QPushButton("📂  إضافة ملفات / مجلد")
        btn_add_files.setStyleSheet("background-color:#2980b9; color:white;")
        btn_add_files.clicked.connect(self.select_files)
        self.btn_add_files = btn_add_files
        bar.addWidget(btn_add_files)

        btn_clear_list = QPushButton("🗑️ تفريغ القائمة")
        btn_clear_list.setStyleSheet("background-color:#e74c3c; color:white;")
        btn_clear_list.clicked.connect(self.clear_file_list)
        self.btn_clear_list = btn_clear_list
        bar.addWidget(btn_clear_list)

        btn_batch_process = QPushButton("🚀 الفحص الشامل للدفعة")
        btn_batch_process.setStyleSheet("background-color:#8e44ad; color:white;")
        btn_batch_process.setEnabled(False)
        btn_batch_process.clicked.connect(self.batch_process_all_files)
        self.btn_batch_process = btn_batch_process
        bar.addWidget(btn_batch_process)

        btn_autocorrect_batch = QPushButton("✨  تصحيح وإدراج ملاحظات (للقائمة كلها)")
        btn_autocorrect_batch.setStyleSheet("background-color:#27ae60; color:white;")
        btn_autocorrect_batch.setEnabled(False)
        btn_autocorrect_batch.clicked.connect(self.auto_correct_and_insert_batch)
        self.btn_autocorrect_batch = btn_autocorrect_batch
        bar.addWidget(btn_autocorrect_batch)

        btn_settings = QPushButton("⚙️  إعدادات الملاحظات")
        btn_settings.setStyleSheet("background-color:#34495e; color:white;")
        btn_settings.clicked.connect(self.open_settings)
        self.btn_settings = btn_settings
        bar.addWidget(btn_settings)
        root.addLayout(bar)

        divider = QLabel(); divider.setFixedHeight(1); divider.setStyleSheet("background:#bdc3c7;")
        root.addWidget(divider)

        body_splitter = QSplitter(Qt.Orientation.Horizontal)
        list_container = QWidget(); list_layout = QVBoxLayout(list_container)
        lbl_list = QLabel("📁 قائمة الملفات:"); list_layout.addWidget(lbl_list)
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemSelectionChanged.connect(self.on_file_selected_from_list)
        list_layout.addWidget(self.file_list_widget)
        body_splitter.addWidget(list_container)

        table_container = QWidget(); table_layout = QVBoxLayout(table_container)
        sel = QHBoxLayout()
        leg = QLabel("نوع الملاحظة:")
        sel.addWidget(leg)
        self.combo_obs_type = QComboBox(); self.combo_obs_type.addItems(("العربية", "الفرنسية", "الانجليزية", "الرياضة"))
        self.combo_obs_type.currentTextChanged.connect(lambda v: setattr(self, "current_subject_type", v))
        sel.addWidget(self.combo_obs_type)
        sel.addSpacing(10)
        txt = QLabel("اللسان (Sheet):"); sel.addWidget(txt)
        self.combo_sheets = QComboBox(); self.combo_sheets.currentTextChanged.connect(self.on_sheet_changed)
        sel.addWidget(self.combo_sheets)
        self.btn_autocorrect_single = QPushButton("✨ تصحيح هدا الملف")
        self.btn_autocorrect_single.setEnabled(False)
        self.btn_autocorrect_single.setStyleSheet("background-color:#f39c12; color:white; padding:6px 10px; font-size:12px;")
        self.btn_autocorrect_single.clicked.connect(self.auto_correct_single)
        sel.addWidget(self.btn_autocorrect_single)
        table_layout.addLayout(sel)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(10)
        self.table.itemChanged.connect(self.on_item_changed)
        table_layout.addWidget(self.table)
        body_splitter.addWidget(table_container)
        body_splitter.setSizes([300, 900])
        root.addWidget(body_splitter)

        info_lb = QLabel("💡 لتصحيح النقطة، اضغط مرتين وعدلها هنا.")
        info_lb.setStyleSheet("color:#2980b9; font-size:11px; font-weight:bold;")
        root.addWidget(info_lb)
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("جاهز. اسحب مجموعة ملفات أو مجلد إلى النافذة لبدء العمل.")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color:#2c3e50; font-size:14px; font-weight:bold;background:#ecf0f1; border:1px solid #bdc3c7; border-radius:6px; padding:10px 12px;")
        status_row.addWidget(self.lbl_status)
        root.addLayout(status_row)

        signature = QLabel('<span style="font-family: Georgia, serif; font-size:18px; font-weight:bold; letter-spacing:2px;">'
                           '<span style="color:#7d3c98;">By: </span><span style="color:#e67e22;">B</span>'
                           '<span style="color:#9b59b6;">@</span><span style="color:#e67e22;">k</span>'
                           '<span style="color:#9b59b6;">h</span><span style="color:#e67e22;">o</span>'
                           '<span style="color:#9b59b6;">u</span><span style="color:#e67e22;">c</span>'
                           '<span style="color:#9b59b6;">h</span><span style="color:#e67e22;">e</span></span>')
        signature.setStyleSheet("background:#ecf0f1; border:1px solid #bdc3c7; border-radius:6px; padding:5px 15px;")
        signature.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(signature)

    def select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "اختر ملفات الإكسال", "", "Excel (*.xlsx *.xls)")
        if paths:
            self.add_files_to_list(paths)

    def add_files_to_list(self, file_paths):
        added = 0
        for p in file_paths:
            if p in self.loaded_files or not os.path.isfile(p):
                continue
            self.loaded_files.append(p)
            name = os.path.basename(p)
            item = QListWidgetItem("📄 " + name)
            item.setToolTip(p)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.file_list_widget.addItem(item)
            added += 1
        self.btn_batch_process.setEnabled(bool(self.loaded_files))
        self.btn_autocorrect_batch.setEnabled(bool(self.loaded_files))
        self.lbl_status.setText(f"تم إضافة {added} ملف جديد لقائمة الانتظار. الإجمالي: {len(self.loaded_files)} ملف.")

    def clear_file_list(self):
        self.loaded_files.clear()
        self.file_list_widget.clear()
        self.table.setRowCount(0); self.table.setColumnCount(0)
        self.combo_sheets.clear()
        self.current_file_path = None
        self.btn_batch_process.setEnabled(False)
        self.btn_autocorrect_batch.setEnabled(False)
        self.btn_autocorrect_single.setEnabled(False)
        self.lbl_status.setText("تم تفريغ القائمة.")

    def on_file_selected_from_list(self):
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        path = item.data(Qt.ItemDataRole.UserRole)
        self.current_file_path = path
        self.load_file_into_view(path)

    def load_file_into_view(self, path):
        try:
            xl = pd.ExcelFile(path)
            valid_sheets = []
            self.excel_data = {}
            for s in xl.sheet_names:
                df = xl.parse(s, header=None)
                self.excel_data[s] = df
                if df.shape[0] > self.DATA_ROW:
                    valid_sheets.append(s)
                errs = self.get_file_errors({s: df})
                has_errors = any(len(v) for v in errs.values())
                if has_errors:
                    self._update_list_item_icon(path, "⚠️ أخطاء", QColor("#e67e22"))
                else:
                    self._update_list_item_icon(path, "✅ سليم", QColor("#27ae60"))
            self.combo_sheets.blockSignals(True)
            self.combo_sheets.clear(); self.combo_sheets.addItems(valid_sheets)
            self.combo_sheets.blockSignals(False)
            self.btn_autocorrect_single.setEnabled(bool(valid_sheets))
            if valid_sheets:
                self.load_sheet(valid_sheets[0])
        except Exception as e:
            QMessageBox.critical(self, "خطأ", "تعذر قراءة الملف " + os.path.basename(path) + ": " + str(e))

    def _update_list_item_icon(self, filepath, text_prefix, color):
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            p = item.data(Qt.ItemDataRole.UserRole)
            if p == filepath:
                name = os.path.basename(filepath)
                item.setText(text_prefix + " | " + name)
                item.setForeground(color)
                return

    def get_file_errors(self, excel_data_dict):
        report = {}
        for sname, df in excel_data_dict.items():
            mark_cols = get_mark_cols(df, self.HEADER_ROW)
            errs = []
            for r in range(self.DATA_ROW, df.shape[0]):
                for c in mark_cols:
                    raw = df.iloc[r, c]
                    if pd.notna(raw):
                        cell_val = str(raw).strip()
                    else:
                        cell_val = ""
                    col_lbl = str(df.iloc[self.HEADER_ROW, c]).strip()
                    if cell_val == "":
                        errs.append(f"السطر {r+1} | عمود {col_lbl}: خانة فارغة")
                        continue
                    if "," in cell_val:
                        errs.append(f"السطر {r+1} | عمود {col_lbl}: فاصلة خاطئة → '{cell_val}'")
                        continue
                    try:
                        v = float(cell_val)
                        if v < 0 or v > 10:
                            errs.append(f"السطر {r+1} | عمود {col_lbl}: قيمة خارج النطاق → {cell_val}")
                    except ValueError:
                        errs.append(f"السطر {r+1} | عمود {col_lbl}: غير رقمي → '{cell_val}'")
            report[sname] = errs
        return report

    def auto_correct_single(self):
        if not self.current_file_path:
            return
        corrections = self._perform_file_auto_correct(self.current_file_path, self.excel_data)
        if corrections:
            self.load_file_into_view(self.current_file_path)
            QMessageBox.information(self, "التصحيح التلقائي", f"تم تصحيح {corrections} خطأ في هذا الملف.")
        else:
            QMessageBox.information(self, "تصحيح", "لم يتم العثور على أخطاء قابلة للتصحيح التلقائي في هذا الملف.")

    def _perform_file_auto_correct(self, path, data_dict):
        wb = openpyxl.load_workbook(path)
        fill_clear = PatternFill(fill_type=None)
        corrections = 0
        for sname in wb.sheetnames:
            ws = wb[sname]
            df = data_dict.get(sname)
            if df is None or df.shape[0] <= self.DATA_ROW:
                continue
            mark_cols = get_mark_cols(df, self.HEADER_ROW)
            for r in range(self.DATA_ROW, df.shape[0]):
                for c in mark_cols:
                    raw_val = ws.cell(row=r + 1, column=c + 1).value
                    if raw_val is None:
                        continue
                    raw_val = str(raw_val).strip()
                    corrected = raw_val.replace(",", ".")
                    if corrected != raw_val:
                        ws.cell(row=r + 1, column=c + 1).value = corrected
                        ws.cell(row=r + 1, column=c + 1).fill = fill_clear
                        corrections += 1
                    elif raw_val and not raw_val.isdigit():
                        try:
                            value = float(raw_val)
                            if 0 <= value <= 10:
                                ws.cell(row=r + 1, column=c + 1).fill = fill_clear
                        except ValueError:
                            pass
        try:
            wb.save(path)
        except PermissionError:
            return 0
        return corrections

    def batch_process_all_files(self):
        self.lbl_status.setText("جارٍ فحص الملفات المحددة...")
        QApplication.processEvents()
        html_lines = ["<h2 style='color:#8e44ad;'>تقرير فحص الدفعة</h2>"]
        text_lines = ["━━ تقرير فحص جميع الملفات ━━\n"]
        total_files = len(self.loaded_files)
        clean_files = 0
        error_files = 0
        for path in self.loaded_files:
            fname = os.path.basename(path)
            html_lines.append("<hr><h3 style='color:#34495e;'>ملف: " + fname + "</h3>")
            text_lines.append("====================\nالملف: " + fname + "\n")
            try:
                xl = pd.ExcelFile(path)
                data_dict = {}
                file_total_errors = 0
                for s in xl.sheet_names:
                    df = xl.parse(s, header=None)
                    data_dict[s] = df
                    if df.shape[0] <= self.DATA_ROW:
                        continue
                    errs = self.get_file_errors({s: df}).get(s, [])
                    sheet_errs = errs
                    file_total_errors += len(sheet_errs)
                    if sheet_errs:
                        html_lines.append(f"<b>اللسان ({s}):</b><ul>" + "".join(f"<li style='color:red;'>{e}</li>" for e in sheet_errs) + "</ul>")
                        text_lines.append(f"  [{s}] " + "\n  ".join(sheet_errs) + "\n")
                if file_total_errors == 0:
                    clean_files += 1
                    self._update_list_item_icon(path, "✅ سليم", QColor("#27ae60"))
                    html_lines.append("<span style='color:green;'>✅ سليم تماماً وبدون أخطاء.</span>")
                    text_lines.append("- سليم تماماً وبدون أخطاء.\n")
                else:
                    error_files += 1
                    self._update_list_item_icon(path, "⚠️ أخطاء", QColor("#e67e22"))
                    html_lines.insert(-1 if html_lines else len(html_lines), f"<span style='color:#e67e22;'>⚠️ عدد الأخطاء: {file_total_errors}</span>")
                    text_lines.append(f"عدد الأخطاء: {file_total_errors}\n")
            except Exception as e:
                error_files += 1
                html_lines.append("<span style='color:red;'>تعذر المعالجة: " + str(e) + "</span>")
                text_lines.append("- تعذر المعالجة: " + str(e) + "\n")
        html_lines.append(f"<h4>تم فحص {total_files} ملفات: <span style='color:green;'>{clean_files} صالحة</span>، <span style='color:red;'>{error_files} بها أخطاء</span>.</h4>")
        text_lines.append(f"فحص شامل مكتمل. {clean_files} ملف سليم | {error_files} ملف به أخطاء.")
        self.lbl_status.setText(f"فحص شامل مكتمل. {clean_files} ملف سليم | {error_files} ملف به أخطاء.")
        dlg = BatchErrorReportDialog(self, "\n".join(html_lines), "\n".join(text_lines))
        dlg.exec()

    def auto_correct_and_insert_batch(self):
        rep = QMessageBox.question(
            self, "تأكيد العملية الجماعية",
            "سوف يقوم البرنامج بالمرور على جميع الملفات الموجودة (" + str(len(self.loaded_files)) +
            " ملف).\n\n1. استبدال جميع الفواصل الخاطئة بنقاط.\n2. إدراج الملاحظات (التقديرات) آلياً.\n3. تخطي الملفات المفتوحة مسبقاً، وتجاهل الخانات المتبقية الفارغة أو الأرقام الكبيرة.\n\nهل متأكد من الاستمرار؟",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if rep != QMessageBox.StandardButton.Yes:
            return
        self.lbl_status.setText("جاري معالجة الدفعة... استرخ، قد يستغرق الأمر بعض الثواني.")
        QApplication.processEvents()
        success_count = 0
        skip_count = 0
        for path in self.loaded_files:
            try:
                xl = pd.ExcelFile(path)
                data_dict = {s: xl.parse(s, header=None) for s in xl.sheet_names}
                self._perform_file_auto_correct(path, data_dict)
                xl_new = pd.ExcelFile(path)
                data_dict_new = {s: xl_new.parse(s, header=None) for s in xl_new.sheet_names}
                self._insert_obs_for_single_file(path, data_dict_new)
                self._update_list_item_icon(path, "✅ مُعالج", QColor("#27ae60"))
                success_count += 1
            except PermissionError:
                self._update_list_item_icon(path, "🔒 مفتوح (تم تخطيه)", QColor("red"))
                skip_count += 1
            except Exception:
                self._update_list_item_icon(path, "❌ خطأ غير متوقع", QColor("red"))
                skip_count += 1
        QMessageBox.information(
            self, "اكتملت العملية الجماعية",
            "تم الانتهاء من المعالجة الجماعية!\n\n✅ تم تصحيح الملاحظات وحفظها في: " + str(success_count) +
            " ملفات.\n⚠️ تم تخطي الملفات (بسبب الإغلاق أو مشاكل): " + str(skip_count) + " ملفات."
        )
        self.lbl_status.setText("انتهت عملية تصحيح الدفعة.")

    def on_sheet_changed(self, name):
        if name:
            self.load_sheet(name)

    def load_sheet(self, name):
        self.current_sheet = name
        df = self.excel_data[name]
        subj_txt = str(name)
        det = detect_subject_type(subj_txt)
        self.combo_obs_type.blockSignals(True)
        self.combo_obs_type.setCurrentText(det)
        self.combo_obs_type.blockSignals(False)
        self.current_subject_type = det
        self._loading = True
        self.table.setRowCount(max(0, df.shape[0] - self.DATA_ROW))
        self.table.setColumnCount(df.shape[1])
        header_labels = [str(df.iloc[self.HEADER_ROW, i]) for i in range(df.shape[1])]
        self.table.setHorizontalHeaderLabels(header_labels)
        mark_cols = get_mark_cols(df, self.HEADER_ROW)
        for r in range(self.DATA_ROW, df.shape[0]):
            for c in range(df.shape[1]):
                raw = df.iloc[r, c]
                text = "" if pd.isna(raw) else str(raw).strip()
                it = QTableWidgetItem(text)
                if c in mark_cols:
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
                    it.setBackground(self._err_color(text))
                else:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                it.setFont(QFont("Arial", 10, QFont.Weight.Bold if c == 0 else QFont.Weight.Normal))
                self.table.setItem(r - self.DATA_ROW, c, it)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._loading = False

    def on_item_changed(self, item):
        if self._loading or not self.current_file_path or not self.current_sheet:
            return
        try:
            df = self.excel_data.get(self.current_sheet)
            c = item.column()
            r_table = item.row()
            r_df = r_table + self.DATA_ROW
            mark_cols = get_mark_cols(df, self.HEADER_ROW)
            if c not in mark_cols:
                return
            val = item.text().strip()
            new_color = self._err_color(val)
            item.setBackground(new_color)
            wb = openpyxl.load_workbook(self.current_file_path)
            ws = wb[self.current_sheet]
            cell = ws.cell(row=r_df + 1, column=c + 1)
            cell.value = val.replace(",", ".") if val else ""
            color_hex = new_color.name() if new_color.isValid() else "white"
            if color_hex != "#ffffff":
                cell.fill = PatternFill(fill_type="solid", fgColor=color_hex.lstrip("#").upper())
            else:
                cell.fill = PatternFill(fill_type=None)
            wb.save(self.current_file_path)
            errs = self.get_file_errors(self.excel_data)
            has_errors = any(len(v) for v in errs.values())
            self._update_list_item_icon(self.current_file_path,
                                        "⚠️ توجد أخطاء" if has_errors else "✅ تم التصحيح بالكامل",
                                        QColor("#e67e22") if has_errors else QColor("#27ae60"))
        except Exception:
            pass

    def _err_color(self, text):
        t = text.strip()
        if t == "":
            return self.color_empty
        if "," in t:
            return self.color_comma
        try:
            v = float(t)
            if v < 0 or v > 10:
                return self.color_range
        except ValueError:
            return self.color_range
        return QColor("white")

    def color_single_file(self, path, file_data_dict):
        try:
            wb = openpyxl.load_workbook(path)
            fill_empty = PatternFill(fill_type="solid", fgColor="FF7675")
            fill_comma = PatternFill(fill_type="solid", fgColor="FFEAA7")
            fill_range = PatternFill(fill_type="solid", fgColor="FAB1A0")
            fill_clear = PatternFill(fill_type=None)
            for sname in wb.sheetnames:
                ws = wb[sname]
                df = file_data_dict.get(sname)
                if df is None:
                    continue
                mcs = get_mark_cols(df, self.HEADER_ROW)
                for row_idx in range(self.DATA_ROW, df.shape[0]):
                    for c in mcs:
                        cell = ws.cell(row=row_idx + 1, column=c + 1)
                        val = "" if cell.value is None else str(cell.value).strip()
                        if val == "":
                            cell.fill = fill_empty
                        elif "," in val:
                            cell.fill = fill_comma
                        else:
                            try:
                                v = float(val)
                                cell.fill = fill_range if v < 0 or v > 10 else fill_clear
                            except ValueError:
                                cell.fill = fill_range
            wb.save(path)
        except Exception:
            return False
        return True

    def _insert_obs_for_single_file(self, path, data_dict):
        wb = openpyxl.load_workbook(path)
        fill_clear = PatternFill(fill_type=None)
        for sname in wb.sheetnames:
            ws = wb[sname]
            df = data_dict.get(sname)
            if df is None or df.shape[0] <= self.DATA_ROW:
                continue
            subj_txt = str(sname)
            det = detect_subject_type(subj_txt)
            rules = self.obs_settings.get(det, [])
            mcs = get_mark_cols(df, self.HEADER_ROW)
            oc = get_obs_col(df, self.HEADER_ROW)
            if oc is None:
                continue
            for r in range(self.DATA_ROW, df.shape[0]):
                xl_row = r + 1
                marks = []
                for c in mcs:
                    cell = ws.cell(row=xl_row, column=c + 1)
                    cell_val = "" if cell.value is None else str(cell.value).strip()
                    if not cell_val:
                        continue
                    cell_val = cell_val.replace(",", ".")
                    try:
                        value = float(cell_val)
                    except Exception:
                        continue
                    if 0 <= value <= 10:
                        marks.append(value)
                if not marks:
                    continue
                avg = sum(marks) / len(marks)
                obs_text = ""
                for lo, hi, txt in rules:
                    if lo <= avg <= hi:
                        obs_text = txt
                        break
                if obs_text:
                    ws.cell(row=xl_row, column=oc + 1).value = obs_text
                    ws.cell(row=xl_row, column=oc + 1).fill = fill_clear
        wb.save(path)
        return True

    def open_settings(self):
        d = ObservationSettingsDialog(self, self.obs_settings)
        if d.exec() == QDialog.DialogCode.Accepted:
            self.obs_settings = d.get_updated_settings()


def main():
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    icon_path = resource_path(os.path.join("img", "001.png"))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    win = ExcelInspectorApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
