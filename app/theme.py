DARK_QSS = """
/* ── Base ─────────────────────────────────────────────────────── */
QWidget {
    background-color: #0F172A;
    color: #E2E8F0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QLabel {
    background: transparent;
    color: #E2E8F0;
}

/* ── Panels / frames ─────────────────────────────────────────── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    color: #334155;
}

/* ── Buttons (base) ──────────────────────────────────────────── */
QPushButton {
    background-color: #1E293B;
    color: #CBD5E1;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover  { background-color: #293548; border-color: #475569; }
QPushButton:pressed { background-color: #0F172A; }
QPushButton:disabled { color: #475569; border-color: #1E293B; background-color: #141E2E; }

/* ── Record buttons (green) ──────────────────────────────────── */
QPushButton#btn_add0,
QPushButton#btn_add1,
QPushButton#btn_add2 {
    background-color: #064E3B;
    border-color: #10B981;
    color: #ECFDF5;
    font-weight: 600;
}
QPushButton#btn_add0:hover,
QPushButton#btn_add1:hover,
QPushButton#btn_add2:hover {
    background-color: #065F46;
}
QPushButton#btn_add0:disabled,
QPushButton#btn_add1:disabled,
QPushButton#btn_add2:disabled {
    background-color: #0A2E22;
    border-color: #1E293B;
    color: #475569;
}

/* ── Cancel button (amber) ───────────────────────────────────── */
QPushButton#btn_cancel {
    background-color: #451A03;
    border-color: #F59E0B;
    color: #FEF3C7;
}
QPushButton#btn_cancel:hover  { background-color: #78350F; }
QPushButton#btn_cancel:disabled { background-color: #1E1406; border-color: #1E293B; color: #475569; }

/* ── Danger button (red) ─────────────────────────────────────── */
QPushButton#btn_danger {
    background-color: #450A0A;
    border-color: #EF4444;
    color: #FEF2F2;
}
QPushButton#btn_danger:hover { background-color: #7F1D1D; }

/* ── Train button (indigo) ───────────────────────────────────── */
QPushButton#btn_train {
    background-color: #1E1B4B;
    border-color: #6366F1;
    color: #EEF2FF;
    font-weight: 700;
    font-size: 14px;
    padding: 8px 14px;
}
QPushButton#btn_train:hover  { background-color: #312E81; border-color: #818CF8; }
QPushButton#btn_train:disabled { background-color: #0F0F2E; border-color: #1E293B; color: #475569; }

/* ── Continue button (accent) ────────────────────────────────── */
QPushButton#btn_continue {
    background-color: #1E1B4B;
    border-color: #6366F1;
    color: #EEF2FF;
    font-weight: 700;
    font-size: 14px;
    padding: 8px 20px;
}
QPushButton#btn_continue:hover   { background-color: #312E81; }
QPushButton#btn_continue:disabled { background-color: #0F0F2E; border-color: #1E293B; color: #475569; }

/* ── Reset (small, neutral) ──────────────────────────────────── */
QPushButton#btn_reset_name {
    background-color: #1E293B;
    color: #94A3B8;
    border-color: #334155;
    font-size: 11px;
    padding: 2px 6px;
}

/* ── Line edits ──────────────────────────────────────────────── */
QLineEdit {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 8px;
    color: #E2E8F0;
    selection-background-color: #6366F1;
}
QLineEdit:focus { border-color: #6366F1; }
QLineEdit:disabled { color: #475569; background-color: #141E2E; }

/* ── Progress bar ────────────────────────────────────────────── */
QProgressBar {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #10B981;
    border-radius: 3px;
}

/* ── Checkbox ────────────────────────────────────────────────── */
QCheckBox { spacing: 8px; color: #CBD5E1; }
QCheckBox::indicator {
    width: 18px; height: 18px;
    border: 2px solid #475569;
    border-radius: 4px;
    background: #1E293B;
}
QCheckBox::indicator:hover   { border-color: #6366F1; }
QCheckBox::indicator:checked {
    background: #6366F1;
    border-color: #6366F1;
}

/* ── Dialogs / MessageBox ────────────────────────────────────── */
QDialog       { background-color: #0F172A; }
QMessageBox   { background-color: #1E293B; }
QMessageBox QLabel { color: #E2E8F0; }
QMessageBox QPushButton { min-width: 80px; }

/* ── ScrollArea ──────────────────────────────────────────────── */
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: #1E293B; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #475569; border-radius: 4px; min-height: 20px;
}
"""
