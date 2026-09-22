# -*- coding: utf-8 -*-
"""Build CustomerPulse_AI_Project_Report.docx — compact, professional college report."""

import json, os
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ASSETS = "report_assets"
OUT    = "CustomerPulse_AI_Project_Report.docx"

with open(f"{ASSETS}/metrics.json") as f:
    M = json.load(f)

S    = M["_summary"]
MED  = M["_medians"]
RISK = M["_risk_counts"]
BEST = M["_best"]
CR   = M["_churn_rate"]

doc = Document()

# ── Page setup (A4, tight margins) ────────────────────────────────────────────
section = doc.sections[0]
section.page_width    = Cm(21)
section.page_height   = Cm(29.7)
section.left_margin   = Cm(2.2)
section.right_margin  = Cm(2.2)
section.top_margin    = Cm(2.0)
section.bottom_margin = Cm(2.0)

BODY_WIDTH_IN = (21 - 2.2 - 2.2) / 2.54   # ≈ 6.52 inches

# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_font(run, bold=False, italic=False, size=11, colour=None, name="Calibri"):
    run.font.name   = name
    run.font.size   = Pt(size)
    run.font.bold   = bold
    run.font.italic = italic
    if colour:
        run.font.color.rgb = RGBColor(*colour)


def _para_fmt(p, space_before=0, space_after=4, line_spacing=None):
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after  = Pt(space_after)
    if line_spacing is not None:
        pf.line_spacing = Pt(line_spacing)


def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    sizes  = {1: 14, 2: 12, 3: 11}
    colors = {1: (0x1a, 0x1a, 0x2e), 2: (0x16, 0x21, 0x3e), 3: (0x1f, 0x23, 0x28)}
    pf = p.paragraph_format
    pf.space_before = Pt(10 if level == 1 else 6)
    pf.space_after  = Pt(3)
    for run in p.runs:
        run.font.name  = "Calibri"
        run.font.size  = Pt(sizes.get(level, 11))
        run.font.color.rgb = RGBColor(*colors.get(level, (0, 0, 0)))
    return p


def para(text="", bold=False, italic=False, size=11,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, colour=None,
         space_before=0, space_after=4):
    p = doc.add_paragraph()
    p.alignment = align
    _para_fmt(p, space_before=space_before, space_after=space_after)
    if text:
        r = p.add_run(text)
        _set_font(r, bold=bold, italic=italic, size=size, colour=colour)
    return p


def bullet(text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    _para_fmt(p, space_before=0, space_after=2)
    r = p.add_run(text)
    _set_font(r, size=size)
    return p


def numbered(text, size=10.5):
    p = doc.add_paragraph(style="List Number")
    _para_fmt(p, space_before=0, space_after=2)
    r = p.add_run(text)
    _set_font(r, size=size)
    return p


def _shade_cell(cell, hex_color="EFF6FF"):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)


def _set_cell_margins(cell, top=40, bottom=40, left=80, right=80):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"),    str(val))
        m.set(qn("w:type"), "dxa")
        tcMar.append(m)
    tcPr.append(tcMar)


def add_table(headers, rows, col_widths=None, header_shade="1E213E",
              alt_shade="F7F8FA", font_size=10):
    """Create a clean, compact table with shaded header and optional alternating rows."""
    ncols = len(headers)
    tbl   = doc.add_table(rows=1 + len(rows), cols=ncols)
    tbl.style     = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        _shade_cell(hdr_cells[i], header_shade)
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _set_cell_margins(hdr_cells[i])
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(0)
        r = p.add_run(h)
        _set_font(r, bold=True, size=font_size, colour=(255, 255, 255))

    # Data rows
    for ri, row_data in enumerate(rows):
        row_cells = tbl.add_row().cells
        shade     = alt_shade if ri % 2 == 1 else "FFFFFF"
        for ci, val in enumerate(row_data):
            _shade_cell(row_cells[ci], shade)
            row_cells[ci].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_margins(row_cells[ci])
            p = row_cells[ci].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after  = Pt(0)
            r = p.add_run(str(val))
            _set_font(r, size=font_size)

    # Column widths
    if col_widths:
        for row in tbl.rows:
            for j, cell in enumerate(row.cells):
                if j < len(col_widths):
                    cell.width = Inches(col_widths[j])

    # Compact paragraph spacing after the table
    p_after = doc.add_paragraph()
    _para_fmt(p_after, space_before=0, space_after=4)
    return tbl


def add_figure(filename, caption_text, width_in=5.8):
    """Add image (if it exists) + italic caption. No extra blank paragraphs."""
    img_path = f"{ASSETS}/{filename}"
    if not img_path.endswith(".png"):
        img_path += ".png"
    if os.path.exists(img_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _para_fmt(p, space_before=4, space_after=2)
        p.add_run().add_picture(img_path, width=Inches(min(width_in, BODY_WIDTH_IN - 0.1)))
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _para_fmt(cp, space_before=0, space_after=6)
    r = cp.add_run(caption_text)
    _set_font(r, italic=True, size=9, colour=(80, 90, 100))


def note_box(text, bg="FFF8E1"):
    """Compact highlighted note paragraph."""
    p = doc.add_paragraph()
    _para_fmt(p, space_before=3, space_after=6)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  bg)
    pPr.append(shd)
    r = p.add_run(text)
    _set_font(r, size=10, italic=True)
    return p


def inline_bold(paragraph, bold_text, normal_text, size=11):
    """Add bold run + normal run to an existing paragraph."""
    r1 = paragraph.add_run(bold_text)
    _set_font(r1, bold=True, size=size)
    r2 = paragraph.add_run(normal_text)
    _set_font(r2, size=size)


def add_page_number():
    """Insert page number field in document footer."""
    section = doc.sections[0]
    footer  = section.footer
    fp      = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    _set_font(run, size=9, colour=(100, 100, 100))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE NUMBERS
# ─────────────────────────────────────────────────────────────────────────────
add_page_number()


# ═════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ═════════════════════════════════════════════════════════════════════════════
for _ in range(4):
    p = doc.add_paragraph()
    _para_fmt(p, space_before=0, space_after=0)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
_para_fmt(p, space_before=0, space_after=2)
r = p.add_run("CustomerPulse AI")
_set_font(r, bold=True, size=30, colour=(26, 33, 62))

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
_para_fmt(p, space_before=0, space_after=6)
r = p.add_run("Automated Customer Churn Prediction & Retention Analytics Platform")
_set_font(r, size=14, colour=(59, 130, 212))

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
_para_fmt(p, space_before=0, space_after=4)
r = p.add_run("Project Technical Report")
_set_font(r, italic=True, size=12, colour=(87, 96, 106))

# Thin separator line
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
_para_fmt(p, space_before=6, space_after=6)
r = p.add_run("─" * 55)
_set_font(r, size=10, colour=(180, 180, 180))

for lbl, val in [
    ("Dataset",       "UCI Online Retail  ·  541,909 transactions  ·  4,334 unique customers"),
    ("Period",        "December 2010 – December 2011"),
    ("Best Model",    "Logistic Regression  ·  CV ROC-AUC = 0.748  ·  Test ROC-AUC = 0.737"),
    ("Tech Stack",    "Python · Streamlit · scikit-learn · pandas · NumPy · matplotlib · seaborn · SHAP"),
]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _para_fmt(p, space_before=0, space_after=2)
    r1 = p.add_run(f"{lbl}:  ")
    _set_font(r1, bold=True, size=10.5, colour=(50, 50, 80))
    r2 = p.add_run(val)
    _set_font(r2, size=10.5, colour=(87, 96, 106))

doc.add_page_break()


# ═════════════════════════════════════════════════════════════════════════════
# 1. ABSTRACT
# ═════════════════════════════════════════════════════════════════════════════
heading("1. Abstract")
para(
    f"CustomerPulse AI is an end-to-end automated analytics platform built with Python and Streamlit "
    f"that transforms raw retail transaction records into actionable customer churn risk intelligence. "
    f"Starting from the UCI Online Retail dataset (541,909 transactions, December 2010 – December 2011), "
    f"the platform automatically cleans the data, engineers 18 customer-level behavioural features "
    f"(RFM components, purchase gap statistics, return behaviour, and last-90-day activity), assigns a "
    f"temporal churn label, and trains four classifiers — Logistic Regression, Random Forest, "
    f"HistGradientBoosting, and K-Nearest Neighbours. The best model is selected by 5-fold "
    f"cross-validated ROC-AUC. Logistic Regression achieved CV ROC-AUC = 0.748 and Test ROC-AUC = 0.737. "
    f"Of the {M['_n_customers']:,} modellable customers, {CR:.1%} are labelled churned. Each customer "
    f"receives a probability score and a HIGH / MEDIUM / LOW risk classification. Predictions are "
    f"explained via SHAP values. Results are surfaced through an interactive multi-page Streamlit app.",
    size=11, space_after=6,
)


# ═════════════════════════════════════════════════════════════════════════════
# 2. INTRODUCTION & OBJECTIVES
# ═════════════════════════════════════════════════════════════════════════════
heading("2. Introduction & Objectives")
para(
    "Customer churn — a customer ceasing to transact with a business — is one of the most commercially "
    "significant risks in retail. Acquiring a new customer typically costs five to seven times more than "
    "retaining an existing one. Early detection of at-risk customers enables targeted retention "
    "interventions (personalised promotions, win-back campaigns) that reduce revenue leakage.",
    size=11,
)
para(
    "CustomerPulse AI addresses this by providing a fully automated, data-driven pipeline that converts "
    "raw transaction records into a ranked list of customers by estimated churn probability, deployable "
    "as a single-command Streamlit application.",
    size=11, space_after=4,
)

heading("Objectives", 2)
for obj in [
    "Automate the complete cleaning, feature engineering, and modelling pipeline.",
    "Engineer 18 interpretable, business-relevant customer-level features.",
    "Train and compare four ML classifiers on a near-balanced churn dataset.",
    "Produce per-customer churn probability scores and HIGH / MEDIUM / LOW risk tiers.",
    "Explain predictions via SHAP values for model transparency.",
    "Deliver results through an interactive Streamlit web application.",
]:
    bullet(obj)


# ═════════════════════════════════════════════════════════════════════════════
# 3. DATASET & PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
heading("3. Dataset & Preprocessing")
para(
    "The project uses the UCI Online Retail dataset — a public, widely-studied transactional dataset "
    "from a UK-based online retailer. Each row is a single line item on an invoice, covering "
    "1 December 2010 to 9 December 2011.",
    size=11,
)

heading("3.1  Dataset Schema", 2)
add_table(
    ["Column", "Type", "Description"],
    [
        ["InvoiceNo",   "String",   "Invoice ID; prefix C = cancellation, A = adjustment"],
        ["StockCode",   "String",   "Product identifier"],
        ["Description", "String",   "Product name (free text)"],
        ["Quantity",    "Integer",  "Units per line item (negative for returns)"],
        ["InvoiceDate", "Datetime", "Invoice timestamp"],
        ["UnitPrice",   "Float",    "Price per unit (£)"],
        ["CustomerID",  "Float",    "Unique customer ID (~25% missing in raw data)"],
        ["Country",     "String",   "Customer's billing country"],
    ],
    col_widths=[1.4, 1.0, 4.1],
)

heading("3.2  Raw Data Statistics", 2)
add_table(
    ["Metric", "Value"],
    [
        ["Total raw rows",                f"{S['original_rows']:,}"],
        ["Date range",                    "2010-12-01  to  2011-12-09"],
        ["Unique countries",              "38"],
        ["Rows missing CustomerID",       f"{S['missing_cid_removed']:,}  (~24.9 %)"],
        ["Cancellation invoices (C-prefix)", f"{S['cancellations_flagged']:,}"],
    ],
    col_widths=[3.2, 3.3],
)

heading("3.3  Automated 13-Step Cleaning Pipeline", 2)
para(
    "The `clean_data()` function applies the following sequential steps automatically on every upload:",
    size=10.5, space_after=3,
)
add_table(
    ["Step", "Operation", "Rows Affected"],
    [
        ["1",  "Drop exact duplicate rows",                           f"{S['duplicates_removed']:,}"],
        ["2",  "Drop rows with missing CustomerID",                   f"{S['missing_cid_removed']:,}"],
        ["3",  "Parse InvoiceDate; drop unparseable",                 f"{S['bad_dates_removed']:,}"],
        ["4",  "Cast Quantity & UnitPrice to numeric; drop failures", "Minimal"],
        ["5",  "Flag invoice type (C = cancel, A = adjust)",          "Flag only"],
        ["6",  "Drop bad-debt adjustment rows (A-prefix)",            "Removed"],
        ["7",  "Separate cancellation rows into returns_df",          f"{S['cancellations_flagged']:,}"],
        ["8",  "Filter to non-cancellation purchase rows only",       "Split"],
        ["9",  "Remove non-product StockCodes (POST, DOT, etc.)",     "Filtered"],
        ["10", "Remove rows with UnitPrice ≤ 0",                      "Filtered"],
        ["11", "Remove rows with Quantity ≤ 0",                       f"{S['non_product_or_invalid']:,} (steps 9–11 combined)"],
        ["12", "Compute LineRevenue = Quantity × UnitPrice",          "Derived column"],
        ["13", "Clean return rows separately (LineRevenue on returns)","Derived column"],
    ],
    col_widths=[0.45, 3.9, 2.15],
    font_size=9.5,
)

p = doc.add_paragraph()
_para_fmt(p, space_before=0, space_after=6)
inline_bold(p, "Final result:  ",
            f"{S['final_purchase_rows']:,} purchase rows  |  {S['unique_customers']:,} unique customers  |  "
            f"Period: {S['date_min'][:10]}  to  {S['date_max'][:10]}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING & CHURN LABEL
# ═════════════════════════════════════════════════════════════════════════════
heading("4. Feature Engineering & Churn Label Definition")
para(
    "A temporal snapshot approach (snapshot date = 2011-09-30) prevents data leakage. All 18 features "
    "are computed from transactions on or before the snapshot. Post-snapshot data is used only for "
    "creating the binary churn label.",
    size=11, space_after=4,
)

heading("4.1  18 Customer-Level Features", 2)
add_table(
    ["Feature", "Description"],
    [
        ["recency_days",            "Days from last purchase to snapshot date"],
        ["frequency",               "Number of distinct purchase dates"],
        ["monetary_total",          "Sum of all line revenues (£)"],
        ["avg_order_value",         "monetary_total ÷ distinct_invoices"],
        ["total_items",             "Total units purchased"],
        ["distinct_products",       "Number of unique products bought"],
        ["distinct_invoices",       "Number of distinct invoices"],
        ["return_rate",             "n_returns ÷ distinct_invoices"],
        ["cancelled_amount",        "Total value of cancelled/returned items (£)"],
        ["n_returns",               "Number of return/cancellation invoices"],
        ["tenure_days",             "Days from first purchase to snapshot"],
        ["purchase_span_days",      "Days from first to last purchase"],
        ["inter_purchase_gap_mean", "Mean days between consecutive purchases"],
        ["inter_purchase_gap_std",  "Std deviation of inter-purchase gaps"],
        ["purchases_last_90d",      "Distinct invoices in 90 days before snapshot"],
        ["revenue_last_90d",        "Revenue in 90 days before snapshot (£)"],
        ["rfm_score",               "Sum of R+F+M quintile scores (range: 3–15)"],
        ["is_uk",                   "Binary flag: 1 = primary country is United Kingdom"],
    ],
    col_widths=[2.2, 4.3],
    font_size=9.5,
)

heading("4.2  RFM Quintile Scoring", 2)
para(
    "Recency, Frequency, and Monetary features are each scored 1–5 using quintile binning. "
    "Lower recency (more recent) → higher R score. Higher frequency/monetary → higher F/M score. "
    "rfm_score = R + F + M (range 3–15). Higher score = more engaged, higher-value customer.",
    size=11, space_after=4,
)

heading("4.3  Churn Label", 2)
para(
    "churn = 1 if the customer made zero qualifying purchases after 2011-09-30 (churned); "
    "churn = 0 if at least one post-snapshot purchase was recorded (retained). "
    "The post-snapshot observation window is ~70 days (Oct 1 – Dec 9, 2011).",
    size=11,
)
note_box(
    f"⚠  The ~70-day observation window (vs. a standard 90-day window) may over-estimate churn. "
    f"Historical churn rate: {CR:.1%}  across  {M['_n_customers']:,} customers."
)


# ═════════════════════════════════════════════════════════════════════════════
# 5. EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
heading("5. Exploratory Data Analysis")
para(
    "Four key visualisations are shown below, selected from the full EDA suite as the most informative "
    "for understanding the dataset and the churn problem.",
    size=11, space_after=4,
)

# --- Figure 1: Churn Distribution ---
heading("Figure 1 — Churn Class Distribution", 2)
churned_n  = int(M["_n_customers"] * CR)
retained_n = M["_n_customers"] - churned_n
add_figure(
    "fig01_churn_distribution",
    f"Fig. 1.  Churn class balance. Churned: {churned_n:,} ({CR:.1%})  |  Retained: {retained_n:,} ({1-CR:.1%}).",
    width_in=4.2,
)
para(
    f"The modelling dataset contains {M['_n_customers']:,} customers. "
    f"{churned_n:,} ({CR:.1%}) are labelled churned and {retained_n:,} ({1-CR:.1%}) retained — "
    "a near-50/50 split caused by the short ~70-day post-snapshot window. "
    "All classifiers use class_weight='balanced' to avoid prediction bias.",
    size=10.5, space_after=4,
)

# --- Figure 2: RFM Scatter ---
heading("Figure 2 — RFM Scatter: Recency vs Frequency by Churn", 2)
add_figure(
    "fig12_rfm_scatter",
    f"Fig. 2.  Recency (days since last purchase) vs Frequency, coloured by churn label. "
    f"Red = churned, green = retained. Sample of up to 1,000 customers shown.",
    width_in=5.6,
)
para(
    "Churned customers (red) cluster in the upper-left — high recency, low frequency — "
    "while retained customers (green) concentrate lower-right. This confirms recency and frequency "
    "as the two strongest individual discriminators for churn.",
    size=10.5, space_after=4,
)

# --- Figure 3: Churn vs Recency + Frequency boxplots side-by-side ---
heading("Figure 3 — Feature Distributions by Churn Status", 2)
add_figure(
    "fig05_churn_vs_recency",
    f"Fig. 3a.  Recency by churn label. Churned median: {MED['recency_churned']:.0f} d  vs  "
    f"Retained: {MED['recency_retained']:.0f} d  (gap = {MED['recency_churned']-MED['recency_retained']:.0f} d).",
    width_in=3.8,
)
add_figure(
    "fig06_churn_vs_frequency",
    f"Fig. 3b.  Frequency by churn label. Retained median: {MED['frequency_retained']:.1f}  vs  "
    f"Churned: {MED['frequency_churned']:.1f}.",
    width_in=3.8,
)
para(
    f"Churned customers show markedly higher recency (median {MED['recency_churned']:.0f} d vs "
    f"{MED['recency_retained']:.0f} d for retained) and lower purchase frequency "
    f"(median {MED['frequency_churned']:.1f} vs {MED['frequency_retained']:.1f}). "
    f"Median lifetime revenue is £{MED['monetary_churned']:,.0f} (churned) vs "
    f"£{MED['monetary_retained']:,.0f} (retained) — a "
    f"{MED['monetary_retained']/MED['monetary_churned']:.1f}× difference.",
    size=10.5, space_after=4,
)

# --- Figure 4: Monthly trends ---
heading("Figure 4 — Monthly Transaction Volume & Revenue", 2)
add_figure(
    "fig09_monthly_trends",
    "Fig. 4.  Monthly invoice volume (left) and revenue (right), Dec 2010 – Dec 2011. "
    "November 2011 is the peak month; December 2011 is a partial month (data ends 9 Dec).",
    width_in=BODY_WIDTH_IN,
)
para(
    "Transaction volume and revenue grew steadily through 2011, peaking in November ahead of the "
    "Christmas season. The partial December 2011 month shows lower volume than the underlying trend. "
    "This seasonal pattern underscores the importance of the snapshot date selection for churn labelling.",
    size=10.5, space_after=4,
)


# ═════════════════════════════════════════════════════════════════════════════
# 6. MODEL TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
heading("6. Model Training & Evaluation")
para(
    f"Train/test split: 80% ({M['_n_train']:,} customers) / 20% ({M['_n_test']:,} customers), "
    "stratified by churn label. All four models are wrapped in a scikit-learn Pipeline with median "
    "imputation and StandardScaler. Model selection criterion: 5-fold stratified CV ROC-AUC.",
    size=11, space_after=4,
)

heading("6.1  Models & Hyperparameters", 2)
add_table(
    ["Model", "Key Hyperparameters"],
    [
        ["Logistic Regression",  "solver=lbfgs, max_iter=1000, class_weight=balanced, random_state=42"],
        ["Random Forest",        "n_estimators=200, class_weight=balanced, random_state=42, n_jobs=-1"],
        ["HistGradientBoosting", "max_iter=200, class_weight=balanced, random_state=42"],
        ["K-Nearest Neighbours", "n_neighbors=7, metric=manhattan (no class weighting)"],
    ],
    col_widths=[2.0, 4.5],
    font_size=10,
)

heading("6.2  Evaluation Metrics — Test Set", 2)
add_table(
    ["Model", "CV ROC-AUC", "Test ROC-AUC", "Accuracy", "Precision", "Recall", "F1", "PR-AUC"],
    [
        [
            n + (" ★" if n == BEST else ""),
            f"{M[n]['cv_roc_auc']:.4f}",
            f"{M[n]['roc_auc']:.4f}",
            f"{M[n]['accuracy']:.4f}",
            f"{M[n]['precision']:.4f}",
            f"{M[n]['recall']:.4f}",
            f"{M[n]['f1']:.4f}",
            f"{M[n]['pr_auc']:.4f}",
        ]
        for n in ["Logistic Regression", "Random Forest",
                  "HistGradientBoosting", "K-Nearest Neighbours"]
    ],
    col_widths=[1.9, 1.05, 1.05, 0.9, 0.9, 0.8, 0.7, 0.85],
    font_size=9.5,
)
note_box(
    f"★ Best model: {BEST}  |  CV ROC-AUC = {M[BEST]['cv_roc_auc']:.4f}  "
    f"|  Test ROC-AUC = {M[BEST]['roc_auc']:.4f}  |  F1 = {M[BEST]['f1']:.4f}"
)

heading("6.3  Model Comparison Chart", 2)
add_figure(
    "fig14_model_comparison",
    "Fig. 5.  CV ROC-AUC (blue) vs Test ROC-AUC (green) for all four models. "
    "Logistic Regression leads on CV ROC-AUC (0.748).",
    width_in=5.8,
)

heading("6.4  Confusion Matrix — Best Model", 2)
add_figure(
    "fig15_confusion_matrix",
    f"Fig. 6.  Confusion matrix for {BEST} on the held-out test set (n = {M['_n_test']:,}).",
    width_in=3.8,
)

heading("6.5  ROC & Precision-Recall Curves", 2)
add_figure(
    "fig16_roc_curves",
    "Fig. 7.  ROC curves for all four models. Best model (★) drawn with a thicker solid line. "
    "Dashed diagonal = random classifier (AUC = 0.50).",
    width_in=5.5,
)
add_figure(
    "fig17_pr_curves",
    "Fig. 8.  Precision-Recall curves for all four models. Higher curve area indicates better "
    "performance on the positive (churned) class.",
    width_in=5.5,
)


# ═════════════════════════════════════════════════════════════════════════════
# 7. CHURN / RISK ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
heading("7. Churn & Risk Analysis")
para(
    "After model selection, the best-model pipeline scores all 3,601 customers. Each customer "
    "receives a continuous churn probability in [0, 1] and is assigned to a risk tier.",
    size=11, space_after=4,
)

heading("7.1  Risk Tier Distribution", 2)
add_table(
    ["Risk Tier", "Probability Range", "Customers", "% of Total", "Colour Code"],
    [
        ["HIGH",   "≥ 0.66",     f"{RISK.get('HIGH', 0):,}",
         f"{100*RISK.get('HIGH', 0)/M['_n_customers']:.1f}%",   "Red   #c0392b"],
        ["MEDIUM", "0.33 – 0.66", f"{RISK.get('MEDIUM', 0):,}",
         f"{100*RISK.get('MEDIUM', 0)/M['_n_customers']:.1f}%", "Amber #e67e22"],
        ["LOW",    "< 0.33",     f"{RISK.get('LOW', 0):,}",
         f"{100*RISK.get('LOW', 0)/M['_n_customers']:.1f}%",    "Green #27ae60"],
    ],
    col_widths=[1.2, 1.5, 1.2, 1.2, 2.0],
)

heading("7.2  Risk Distribution Chart", 2)
add_figure(
    "fig19_risk_distribution",
    f"Fig. 9.  Predicted risk distribution across all {M['_n_customers']:,} customers. "
    f"HIGH: {RISK.get('HIGH', 0):,}  |  MEDIUM: {RISK.get('MEDIUM', 0):,}  |  LOW: {RISK.get('LOW', 0):,}.",
    width_in=4.5,
)

heading("7.3  Feature Importances", 2)
add_figure(
    "fig18_feature_importance",
    f"Fig. 10.  Top feature importances from {BEST}. Recency-related features dominate, "
    "followed by frequency and last-90-day activity.",
    width_in=5.8,
)
para(
    "Recency-related features (recency_days, rfm_score) dominate the model's discriminatory power, "
    "followed by frequency metrics (purchases_last_90d, frequency). Return behaviour (return_rate) "
    "also contributes, indicating that customers with a higher return proportion are at elevated risk.",
    size=10.5, space_after=4,
)


# ═════════════════════════════════════════════════════════════════════════════
# 8. KEY FINDINGS & BUSINESS INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
heading("8. Key Findings & Business Insights")
para(
    "The following four findings are derived directly from the actual computed results — "
    "they are observations from the data, not causal claims.",
    size=11, space_after=4,
)

insights = [
    (
        "Finding 1 — Recency Is the Strongest Churn Signal",
        f"Churned customers last purchased a median of {MED['recency_churned']:.0f} days before the "
        f"snapshot vs. {MED['recency_retained']:.0f} days for retained — a gap of "
        f"{MED['recency_churned']-MED['recency_retained']:.0f} days. Recency is also the top-ranked "
        "feature by model importance. Retailers should prioritise win-back campaigns targeting "
        "customers whose last purchase was more than ~60–70 days ago.",
    ),
    (
        "Finding 2 — Frequency Separates Engaged From At-Risk Customers",
        f"Retained customers had a median of {MED['frequency_retained']:.1f} distinct purchase dates "
        f"vs. {MED['frequency_churned']:.1f} for churned — a 3× difference. Customers who have made "
        "only a single purchase are disproportionately likely to churn. Incentivising a second purchase "
        "within 30 days of the first (e.g. loyalty discount) could meaningfully improve retention.",
    ),
    (
        "Finding 3 — Revenue Is Significantly Higher for Retained Customers",
        f"Median lifetime revenue: £{MED['monetary_retained']:,.0f} (retained) vs. "
        f"£{MED['monetary_churned']:,.0f} (churned) — a "
        f"{MED['monetary_retained']/MED['monetary_churned']:.1f}× multiple. "
        "High-monetary customers who have not transacted in 90+ days represent the highest-value "
        "churn risk and should be the first target for personalised retention outreach.",
    ),
    (
        f"Finding 4 — {RISK.get('HIGH',0):,} Customers Are in the High-Risk Tier",
        f"The model assigns {RISK.get('HIGH',0):,} customers ({100*RISK.get('HIGH',0)/M['_n_customers']:.1f}%) "
        f"to HIGH risk (probability ≥ 0.66) and {RISK.get('MEDIUM',0):,} to MEDIUM risk. "
        "The Risk Explorer enables teams to export and act on this prioritised list directly, "
        "enabling targeted campaigns rather than broad-brush interventions.",
    ),
]

for title, body in insights:
    p = doc.add_paragraph()
    _para_fmt(p, space_before=6, space_after=2)
    inline_bold(p, title + "  —  ", body)


# ═════════════════════════════════════════════════════════════════════════════
# 9. CONCLUSION
# ═════════════════════════════════════════════════════════════════════════════
heading("9. Conclusion")
para(
    "CustomerPulse AI demonstrates that a well-structured, single-file Python application can deliver "
    "a complete, production-quality churn prediction pipeline — from raw transaction data to "
    "per-customer risk scores and SHAP explanations — with no manual intervention beyond uploading a CSV.",
    size=11,
)
para(
    f"Applied to the UCI Online Retail dataset ({S['original_rows']:,} rows, {S['unique_customers']:,} customers), "
    f"the platform models {M['_n_customers']:,} customers and finds that {CR:.1%} churned in the "
    f"~70-day post-snapshot window. Logistic Regression (CV ROC-AUC = 0.748, Test ROC-AUC = 0.737) "
    "is selected as the best model, achieving genuine discriminatory power on this near-balanced dataset.",
    size=11,
)
para(
    f"The Risk Explorer surfaces {RISK.get('HIGH',0):,} HIGH-risk and {RISK.get('MEDIUM',0):,} MEDIUM-risk "
    "customers as immediately actionable retention targets. The platform's key finding is that recency "
    f"is overwhelmingly the strongest churn predictor: churned customers last transacted a median of "
    f"{MED['recency_churned']:.0f} days before the snapshot vs. {MED['recency_retained']:.0f} days for retained.",
    size=11, space_after=4,
)


# ═════════════════════════════════════════════════════════════════════════════
# 10. FUTURE SCOPE
# ═════════════════════════════════════════════════════════════════════════════
heading("10. Future Scope")
for f in [
    "Configurable snapshot date and churn window via the Streamlit sidebar (currently hard-coded to 2011-09-30).",
    "Hyperparameter optimisation (GridSearchCV / RandomizedSearchCV) to improve HistGradientBoosting performance.",
    "Survival analysis (Cox proportional hazards) for time-to-churn modelling instead of binary labelling.",
    "Customer Lifetime Value (CLV) integration to weight retention decisions by predicted revenue impact.",
    "Cohort analysis: visualise churn rates by acquisition month and country.",
    "CSV export of the full risk table, filtered by tier, for direct CRM / email campaign ingestion.",
    "Automated re-training pipeline triggered by new data uploads.",
    "Calibration curves and confidence intervals for churn probability estimates.",
]:
    bullet(f)


# ═════════════════════════════════════════════════════════════════════════════
# REFERENCES
# ═════════════════════════════════════════════════════════════════════════════
heading("References")
refs = [
    "Chen, D., Sain, S.L. & Guo, K. (2012). Data mining for the online retail industry: A case study "
    "of RFM model-based customer segmentation. Journal of Database Marketing & Customer Strategy "
    "Management, 19(3), 197–208.",

    "UCI Machine Learning Repository. Online Retail Dataset. Dua, D. and Graff, C. (2019). "
    "UCI Machine Learning Repository. Irvine, CA: University of California.",

    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. "
    "Journal of Machine Learning Research, 12, 2825–2830.",

    "Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. "
    "Advances in Neural Information Processing Systems, 30.",

    "Streamlit Inc. (2024). Streamlit — The fastest way to build and share data apps. "
    "https://streamlit.io",
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    _para_fmt(p, space_before=0, space_after=3)
    r = p.add_run(f"[{i}]  {ref}")
    _set_font(r, size=10)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
doc.save(OUT)
print(f"Saved: {OUT}")
