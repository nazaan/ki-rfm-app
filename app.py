"""
app.py
------
KI DataLab — RFM Customer Segmentation Tool
Main Streamlit UI. Pure presentation layer — no business logic here.

All calculations, charts, exports and AI live in their respective modules.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date

# ── Auth ───────────────────────────────────────────────────────────────────
from auth import (
    require_subscription,
    handle_token_from_url,
    show_auth_status_sidebar,
    is_agency,
)

# ── Business logic ─────────────────────────────────────────────────────────
from utils.data_loader import load_and_validate, get_data_summary
from calculations.rfm_engine import (
    compute_rfm,
    build_rfm_matrix,
    get_matrix_cell_customers,
    recompute_weighted_total,
)
from calculations.segments import assign_segments, get_segment_summary
from calculations.stats import (
    compute_quick_stats,
    compute_health_score,
    fmt_currency,
    fmt_pct,
    fmt_number,
)
from charts.bar_chart import build_bar_chart
from charts.donut_chart import build_donut_chart
from charts.scatter_chart import build_scatter_chart
from exports.csv_export import (
    export_full_rfm,
    export_segment,
    export_vip_list,
    export_churn_risk,
    export_winback_campaign,
    export_klaviyo,
    export_mailchimp,
)
from ai.llm_router import LLMRouter, PROVIDERS, get_quality_note, LLMError
from ai.prompts import (
    monday_morning_brief_prompt,
    segment_persona_prompt,
    winback_subject_lines_prompt,
    churn_explanation_prompt,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RFM Analyser — KI DataLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colors ───────────────────────────────────────────────────────────
GREEN1   = "#10bb82"
GREEN2   = "#009084"
PURPLE   = "#561269"
CHARCOAL = "#1A1A2E"
WHITE    = "#FFFFFF"
GRAY_LT  = "#EEF2F0"
GRAY_MID = "#C8D4CE"
GRAY_DRK = "#6B7E75"

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Base ── */
  [data-testid="stAppViewContainer"] {{
      background: #F8FAFB;
  }}
  [data-testid="stSidebar"] {{
      background: {CHARCOAL};
  }}
  [data-testid="stSidebar"] * {{
      color: {WHITE} !important;
  }}
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stDateInput label,
  [data-testid="stSidebar"] .stTextInput label {{
      color: {GRAY_MID} !important;
      font-size: 12px !important;
  }}

  /* ── KPI tiles ── */
  .kpi-tile {{
      background: {WHITE};
      border-radius: 10px;
      padding: 16px 14px 12px 14px;
      border-left: 4px solid {GREEN1};
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .kpi-value {{
      font-size: 22px;
      font-weight: 700;
      color: {CHARCOAL};
      line-height: 1.2;
  }}
  .kpi-label {{
      font-size: 11px;
      color: {GRAY_DRK};
      margin-top: 2px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
  }}

  /* ── Section headers ── */
  .section-header {{
      font-size: 13px;
      font-weight: 700;
      color: {GRAY_DRK};
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 0 0 8px 0;
      border-bottom: 2px solid {GREEN1};
      margin-bottom: 16px;
  }}

  /* ── Segment badge ── */
  .seg-badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
  }}

  /* ── Agency badge ── */
  .agency-badge {{
      display: inline-block;
      background: {PURPLE};
      color: white;
      padding: 2px 8px;
      border-radius: 10px;
      font-size: 10px;
      font-weight: 600;
      margin-left: 6px;
      vertical-align: middle;
  }}

  /* ── Health score ── */
  .health-block {{
      background: {WHITE};
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
      text-align: center;
  }}
  .health-score {{
      font-size: 48px;
      font-weight: 800;
      line-height: 1;
  }}
  .health-verdict {{
      font-size: 14px;
      font-weight: 600;
      margin-top: 4px;
  }}

  /* ── Matrix cell ── */
  .matrix-cell {{
      background: {WHITE};
      border: 1px solid {GRAY_MID};
      border-radius: 6px;
      padding: 8px 4px;
      text-align: center;
      cursor: pointer;
      font-size: 18px;
      font-weight: 700;
  }}

  /* ── Action card ── */
  .action-card {{
      background: {WHITE};
      border-radius: 10px;
      padding: 16px;
      border-top: 3px solid {GREEN1};
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}

  /* ── Brief box ── */
  .brief-box {{
      background: {WHITE};
      border-left: 4px solid {GREEN1};
      border-radius: 8px;
      padding: 20px 24px;
      font-size: 14px;
      line-height: 1.7;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {{
      border: 2px dashed {GREEN1} !important;
      border-radius: 10px !important;
      background: rgba(16,187,130,0.04) !important;
  }}

  /* ── Divider ── */
  hr {{ border-color: {GRAY_LT}; }}

  /* ── Table ── */
  .rfm-table th {{
      background: {CHARCOAL};
      color: {WHITE};
      padding: 8px 12px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
  }}
  .rfm-table td {{
      padding: 7px 12px;
      font-size: 12px;
      border-bottom: 1px solid {GRAY_LT};
  }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HANDLE MAGIC LINK TOKEN (must be first)
# ══════════════════════════════════════════════════════════════════════════
handle_token_from_url()

# ══════════════════════════════════════════════════════════════════════════
# AUTH GATE
# ══════════════════════════════════════════════════════════════════════════
user = require_subscription(min_tier="small_biz")
agency = is_agency(user)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand
    st.markdown(f"""
        <div style="padding:8px 0 20px 0">
          <div style="font-size:20px;font-weight:800;color:{GREEN1};
                      letter-spacing:-0.5px">KI DataLab</div>
          <div style="font-size:11px;color:{GRAY_MID};margin-top:2px">
            RFM Customer Segmentation
          </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Upload ─────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header" style="color:{GRAY_MID}">DATA</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload orders CSV",
        type=["csv"],
        help="Shopify, Etsy, WooCommerce or any CSV with Customer ID, Date, Order Value",
        label_visibility="collapsed",
    )
    st.caption("Supports Shopify · Etsy · WooCommerce · generic CSV")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Settings ────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header" style="color:{GRAY_MID}">SETTINGS</div>',
                unsafe_allow_html=True)

    reference_date = st.date_input(
        "Reference date",
        value=date.today(),
        help="Recency is calculated from this date. Default: today.",
    )

    churn_days = st.slider(
        "Churn threshold (days)",
        min_value=30, max_value=365, value=90, step=10,
        help="Customers inactive longer than this are counted as churned.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RFM weights ─────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header" style="color:{GRAY_MID}">RFM WEIGHTS</div>',
                unsafe_allow_html=True)
    st.caption("Adjust importance of each dimension")

    r_weight = st.slider("Recency (R)",   0.5, 3.0, 1.0, 0.5, key="r_w")
    f_weight = st.slider("Frequency (F)", 0.5, 3.0, 1.0, 0.5, key="f_w")
    m_weight = st.slider("Monetary (M)",  0.5, 3.0, 1.0, 0.5, key="m_w")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI provider ─────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header" style="color:{GRAY_MID}">AI SETTINGS</div>',
                unsafe_allow_html=True)

    provider_options = list(PROVIDERS.keys())
    provider_labels  = [PROVIDERS[p]["label"] for p in provider_options]

    selected_provider = st.selectbox(
        "AI Provider",
        options=provider_options,
        format_func=lambda p: PROVIDERS[p]["label"],
        key="ai_provider",
    )

    quality_note = get_quality_note(selected_provider)
    if quality_note:
        st.caption(quality_note)

    needs_key = PROVIDERS[selected_provider]["needs_key"]
    api_key   = ""

    if needs_key:
        api_key = st.text_input(
            PROVIDERS[selected_provider]["key_label"],
            type="password",
            placeholder=PROVIDERS[selected_provider]["key_prefix"] + "...",
            help=PROVIDERS[selected_provider]["note"],
            key="ai_key",
        )
    else:
        ollama_url = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            key="ollama_url",
        )
        st.caption(PROVIDERS[selected_provider]["note"])

    # ── Auth status ─────────────────────────────────────────────────────
    show_auth_status_sidebar(user)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
  <div style="display:flex;align-items:center;justify-content:space-between;
              padding:16px 0 8px 0;border-bottom:2px solid {GREEN1};
              margin-bottom:24px">
    <div>
      <span style="font-size:26px;font-weight:800;color:{CHARCOAL}">
        RFM Customer Segmentation
      </span>
      <span style="font-size:13px;color:{GRAY_DRK};margin-left:12px">
        by KI DataLab
      </span>
    </div>
    <div style="font-size:11px;color:{GRAY_MID}">
      kidatalab.com
    </div>
  </div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# NO FILE UPLOADED — LANDING STATE
# ══════════════════════════════════════════════════════════════════════════
if uploaded_file is None:
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(f"""
          <div style="text-align:center;padding:60px 20px">
            <div style="font-size:48px">📊</div>
            <div style="font-size:22px;font-weight:700;color:{CHARCOAL};
                        margin:16px 0 8px 0">
              Upload your orders CSV to get started
            </div>
            <div style="font-size:14px;color:{GRAY_DRK};line-height:1.6">
              Drag and drop your CSV into the sidebar.<br>
              Works with Shopify, Etsy, WooCommerce, or any orders export.<br>
              <strong>Your data never leaves your session.</strong>
            </div>
            <div style="margin-top:32px;padding:16px 24px;
                        background:{GRAY_LT};border-radius:10px;
                        font-size:12px;color:{GRAY_DRK};text-align:left">
              <strong>Required columns:</strong> Customer ID · Order Date · Order Value<br>
              <strong>Optional:</strong> Customer Name<br>
              <strong>Formats:</strong> YYYY-MM-DD · DD/MM/YYYY · MM/DD/YYYY
            </div>
          </div>
        """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# LOAD + PROCESS DATA
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def process_data(file_bytes: bytes, ref_date: date, r_w: float, f_w: float, m_w: float):
    import io
    file = io.BytesIO(file_bytes)
    df, warnings = load_and_validate(file)
    ref_dt = datetime.combine(ref_date, datetime.min.time())
    rfm    = compute_rfm(df, reference_date=ref_dt,
                         recency_weight=r_w,
                         frequency_weight=f_w,
                         monetary_weight=m_w)
    rfm    = assign_segments(rfm)
    return df, rfm, warnings


with st.spinner("Processing your data…"):
    try:
        file_bytes = uploaded_file.read()
        df, rfm, load_warnings = process_data(
            file_bytes, reference_date, r_weight, f_weight, m_weight
        )
    except ValueError as e:
        st.error(f"**Could not load file:** {e}")
        st.stop()

# Show non-fatal warnings collapsed
if load_warnings:
    with st.expander(f"⚠️  {len(load_warnings)} data warning(s) — click to review"):
        for w in load_warnings:
            st.caption(f"• {w}")

# Downstream calculations
summary = get_segment_summary(rfm)
stats   = compute_quick_stats(df, rfm, churn_days=churn_days,
                               reference_date=datetime.combine(reference_date, datetime.min.time()))
health  = compute_health_score(stats, summary)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — QUICK STATS
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)

kpi_defs = [
    ("Total Customers",      fmt_number(stats["total_customers"]),     GREEN1),
    ("Total Revenue",        fmt_currency(stats["total_revenue"]),     GREEN2),
    ("Avg Order Value",      fmt_currency(stats["avg_order_value"]),   GREEN1),
    ("Repeat Purchase Rate", fmt_pct(stats["repeat_purchase_rate"]),   GREEN2),
    ("Churn Rate",           fmt_pct(stats["churn_rate"]),             "#D46A17"),
    ("Top 5% Revenue Share", fmt_pct(stats["top5_revenue_pct"]),       GREEN1),
    ("Avg Orders/Customer",  str(stats["avg_orders_per_customer"]),    GREEN2),
    ("Date Range",           f"{stats['date_range_start']} → {stats['date_range_end']}", CHARCOAL),
]

kpi_cols = st.columns(4)
for i, (label, value, accent) in enumerate(kpi_defs):
    with kpi_cols[i % 4]:
        st.markdown(f"""
          <div class="kpi-tile" style="border-left-color:{accent}">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
          </div>
          <div style="height:10px"></div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — HEALTH SCORE + MONDAY MORNING BRIEF
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">Health Score & Monday Morning Brief</div>',
            unsafe_allow_html=True)

col_health, col_brief = st.columns([1, 3])

with col_health:
    st.markdown(f"""
      <div class="health-block">
        <div class="health-score" style="color:{health['color']}">
          {health['score']}
        </div>
        <div class="health-verdict" style="color:{health['color']}">
          {health['verdict']}
        </div>
        <div style="font-size:10px;color:{GRAY_DRK};margin-top:8px">
          Customer Base Health
        </div>
      </div>
    """, unsafe_allow_html=True)

with col_brief:
    from calculations.segments import get_high_value_at_risk
    at_risk = get_high_value_at_risk(rfm)

    ai_ready = bool(api_key) or selected_provider == "ollama"

    if not ai_ready:
        st.info(
            "Enter your API key in the sidebar to generate the Monday Morning Brief.",
            icon="🤖",
        )
    else:
        if st.button("📋  Generate Monday Morning Brief", type="primary", key="brief_btn"):
            with st.spinner("Generating brief…"):
                try:
                    router = LLMRouter(
                        provider=selected_provider,
                        api_key=api_key,
                        ollama_url=st.session_state.get("ollama_url", "http://localhost:11434"),
                    )
                    sys_p, usr_p = monday_morning_brief_prompt(stats, summary, health, at_risk)
                    brief_text   = router.complete(sys_p, usr_p, max_tokens=400)
                    st.session_state["brief_text"] = brief_text
                except LLMError as e:
                    st.error(f"AI error: {e}")

    if "brief_text" in st.session_state:
        st.markdown(
            f'<div class="brief-box">{st.session_state["brief_text"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CHARTS
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">Charts</div>', unsafe_allow_html=True)

col_bar, col_donut = st.columns(2)

with col_bar:
    st.plotly_chart(build_bar_chart(summary), use_container_width=True)

with col_donut:
    st.plotly_chart(build_donut_chart(summary), use_container_width=True)

st.plotly_chart(build_scatter_chart(summary), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — 5x5 RFM MATRIX
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">RFM Matrix — Recency vs Frequency</div>',
            unsafe_allow_html=True)
st.caption("Click any cell to see the customers in that group.")

matrix = build_rfm_matrix(rfm)

# Column headers
header_cols = st.columns([0.6] + [1] * 5)
with header_cols[0]:
    st.markdown(
        f'<div style="text-align:center;font-size:10px;color:{GRAY_DRK};'
        f'padding-top:8px">R \\ F →</div>',
        unsafe_allow_html=True,
    )
for i, f_val in enumerate([1, 2, 3, 4, 5]):
    with header_cols[i + 1]:
        st.markdown(
            f'<div style="text-align:center;font-size:11px;font-weight:700;'
            f'color:{GREEN2};padding:4px 0">F={f_val}</div>',
            unsafe_allow_html=True,
        )

# Matrix rows
for r_val in [5, 4, 3, 2, 1]:
    row_cols = st.columns([0.6] + [1] * 5)
    with row_cols[0]:
        st.markdown(
            f'<div style="text-align:right;font-size:11px;font-weight:700;'
            f'color:{GREEN1};padding:12px 8px 0 0">R={r_val}</div>',
            unsafe_allow_html=True,
        )
    for f_val in [1, 2, 3, 4, 5]:
        count = int(matrix.loc[r_val, f_val]) if (r_val in matrix.index and f_val in matrix.columns) else 0
        with row_cols[f_val]:
            # Color intensity based on count
            if count == 0:
                bg, fg = GRAY_LT, GRAY_DRK
            elif r_val >= 4 and f_val >= 4:
                bg, fg = GREEN1, WHITE
            elif r_val >= 3 and f_val >= 3:
                bg, fg = GREEN2, WHITE
            elif r_val <= 2 and f_val <= 2:
                bg, fg = "#B03A2E", WHITE
            else:
                bg, fg = "#E8A838", CHARCOAL

            btn_key = f"matrix_{r_val}_{f_val}"
            if st.button(
                str(count),
                key=btn_key,
                help=f"R={r_val}, F={f_val}: {count} customer(s)",
                use_container_width=True,
            ):
                st.session_state["matrix_selected"] = (r_val, f_val)

# Show selected cell customers in expander
if "matrix_selected" in st.session_state:
    r_sel, f_sel = st.session_state["matrix_selected"]
    cell_customers = get_matrix_cell_customers(rfm, r_sel, f_sel)
    label = f"R={r_sel}, F={f_sel} — {len(cell_customers)} customer(s)"
    with st.expander(f"👥  {label}", expanded=True):
        if cell_customers.empty:
            st.caption("No customers in this cell.")
        else:
            display_cols = {
                "customer_id":    "Customer ID",
                "customer_name":  "Name",
                "r_score":        "R",
                "f_score":        "F",
                "m_score":        "M",
                "rfm_total":      "RFM",
                "days_since":     "Days Since",
                "order_count":    "Orders",
                "total_spend":    "Total Spend",
            }
            show_cols = [c for c in display_cols if c in cell_customers.columns]
            display_df = cell_customers[show_cols].rename(columns=display_cols)
            if "Total Spend" in display_df.columns:
                display_df["Total Spend"] = display_df["Total Spend"].apply(
                    lambda x: f"${x:,.2f}"
                )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — SEGMENT BREAKDOWN + ACTION CARDS
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">Segment Breakdown & Exports</div>',
            unsafe_allow_html=True)

# ── Full export button ─────────────────────────────────────────────────
col_full_l, col_full_r = st.columns([3, 1])
with col_full_r:
    full_bytes, full_fname = export_full_rfm(rfm)
    st.download_button(
        "⬇  Download Full RFM Export",
        data=full_bytes,
        file_name=full_fname,
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Segment rows ───────────────────────────────────────────────────────
for _, seg_row in summary.iterrows():
    seg_name = seg_row["segment"]
    count    = int(seg_row["count"])
    bg_color = seg_row["color"]
    fg_color = seg_row["text"]

    # Container per segment
    with st.container():
        col_badge, col_stats, col_actions = st.columns([2, 3, 4])

        # Badge + count
        with col_badge:
            st.markdown(f"""
              <div style="display:flex;align-items:center;gap:10px;padding:10px 0">
                <span class="seg-badge" style="background:{bg_color};color:{fg_color}">
                  {seg_name}
                </span>
                <span style="font-size:20px;font-weight:700;color:{CHARCOAL}">
                  {count}
                </span>
                <span style="font-size:11px;color:{GRAY_DRK}">
                  ({seg_row['pct_customers']}%)
                </span>
              </div>
            """, unsafe_allow_html=True)

        # Stats
        with col_stats:
            if count > 0:
                st.markdown(f"""
                  <div style="padding:10px 0;font-size:12px;color:{GRAY_DRK};
                              display:flex;gap:20px;flex-wrap:wrap">
                    <span>Avg Spend <strong style="color:{CHARCOAL}">
                      {fmt_currency(seg_row['avg_spend'])}</strong></span>
                    <span>Revenue Share <strong style="color:{CHARCOAL}">
                      {fmt_pct(seg_row['pct_revenue'])}</strong></span>
                    <span>Avg RFM <strong style="color:{CHARCOAL}">
                      {seg_row['avg_rfm']:.1f}</strong></span>
                  </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="padding:10px 0;font-size:12px;color:{GRAY_MID}">'
                    f'No customers in this segment</div>',
                    unsafe_allow_html=True,
                )

        # Action buttons
        with col_actions:
            if count > 0:
                btn_cols = st.columns(4)

                # CSV export
                with btn_cols[0]:
                    seg_bytes, seg_fname = export_segment(rfm, seg_name)
                    st.download_button(
                        "CSV",
                        data=seg_bytes,
                        file_name=seg_fname,
                        mime="text/csv",
                        key=f"dl_seg_{seg_name}",
                        help=f"Download {seg_name} customers as CSV",
                        use_container_width=True,
                    )

                # Klaviyo export
                with btn_cols[1]:
                    kl_bytes, kl_fname = export_klaviyo(rfm, seg_name)
                    st.download_button(
                        "Klaviyo",
                        data=kl_bytes,
                        file_name=kl_fname,
                        mime="text/csv",
                        key=f"dl_kl_{seg_name}",
                        help=f"Download {seg_name} in Klaviyo import format",
                        use_container_width=True,
                    )

                # Mailchimp export
                with btn_cols[2]:
                    mc_bytes, mc_fname = export_mailchimp(rfm, seg_name)
                    st.download_button(
                        "Mailchimp",
                        data=mc_bytes,
                        file_name=mc_fname,
                        mime="text/csv",
                        key=f"dl_mc_{seg_name}",
                        help=f"Download {seg_name} in Mailchimp import format",
                        use_container_width=True,
                    )

                # AI persona (agency only)
                with btn_cols[3]:
                    if agency:
                        if st.button(
                            "Persona",
                            key=f"persona_{seg_name}",
                            help="Generate AI marketing persona for this segment",
                            use_container_width=True,
                        ):
                            st.session_state["persona_target"] = seg_name
                    else:
                        st.markdown(
                            '<span class="agency-badge">Agency</span>',
                            unsafe_allow_html=True,
                        )

        # Segment action tip
        st.caption(f"💡  {seg_row['action']}")

        # AI Persona output (agency only)
        if agency and st.session_state.get("persona_target") == seg_name:
            if not ai_ready:
                st.info("Enter your API key in the sidebar to generate a persona.", icon="🤖")
            else:
                persona_key = f"persona_text_{seg_name}"
                if persona_key not in st.session_state:
                    with st.spinner(f"Writing persona for {seg_name}…"):
                        try:
                            router = LLMRouter(
                                provider=selected_provider,
                                api_key=api_key,
                                ollama_url=st.session_state.get("ollama_url","http://localhost:11434"),
                            )
                            seg_customers = rfm[rfm["segment"] == seg_name]
                            sys_p, usr_p  = segment_persona_prompt(seg_name, seg_row, seg_customers)
                            persona_text  = router.complete(sys_p, usr_p, max_tokens=300)
                            st.session_state[persona_key] = persona_text
                        except LLMError as e:
                            st.error(f"AI error: {e}")

                if persona_key in st.session_state:
                    with st.expander(f"🎯  {seg_name} Persona", expanded=True):
                        st.markdown(st.session_state[persona_key])

        st.markdown(f'<hr style="margin:8px 0;border-color:{GRAY_LT}">', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — ACTION CARDS
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)

ac1, ac2, ac3, ac4 = st.columns(4)

with ac1:
    st.markdown(f"""
      <div class="action-card">
        <div style="font-size:13px;font-weight:700;color:{CHARCOAL};margin-bottom:6px">
          🏆 VIP List for Meta Ads
        </div>
        <div style="font-size:11px;color:{GRAY_DRK};margin-bottom:12px">
          Champions + Loyal Customers.<br>
          Ready to upload as a custom audience.
        </div>
      </div>
    """, unsafe_allow_html=True)
    vip_bytes, vip_fname = export_vip_list(rfm)
    st.download_button(
        "Download VIP List",
        data=vip_bytes,
        file_name=vip_fname,
        mime="text/csv",
        key="dl_vip",
        use_container_width=True,
    )

with ac2:
    st.markdown(f"""
      <div class="action-card">
        <div style="font-size:13px;font-weight:700;color:{CHARCOAL};margin-bottom:6px">
          🔴 Churn Risk Export
        </div>
        <div style="font-size:11px;color:{GRAY_DRK};margin-bottom:12px">
          At Risk + Cannot Lose customers.<br>
          Prioritised by lifetime spend.
        </div>
      </div>
    """, unsafe_allow_html=True)
    cr_bytes, cr_fname = export_churn_risk(rfm)
    st.download_button(
        "Export Churn Risk",
        data=cr_bytes,
        file_name=cr_fname,
        mime="text/csv",
        key="dl_churn",
        use_container_width=True,
    )

with ac3:
    st.markdown(f"""
      <div class="action-card">
        <div style="font-size:13px;font-weight:700;color:{CHARCOAL};margin-bottom:6px">
          📧 Win-back Campaign
        </div>
        <div style="font-size:11px;color:{GRAY_DRK};margin-bottom:12px">
          At Risk + Cannot Lose + About To Sleep.<br>
          Full re-engagement list.
        </div>
      </div>
    """, unsafe_allow_html=True)
    wb_bytes, wb_fname = export_winback_campaign(rfm)
    st.download_button(
        "Generate Win-back List",
        data=wb_bytes,
        file_name=wb_fname,
        mime="text/csv",
        key="dl_winback",
        use_container_width=True,
    )

with ac4:
    # Win-back subject lines — agency only
    st.markdown(f"""
      <div class="action-card" style="border-top-color:{PURPLE}">
        <div style="font-size:13px;font-weight:700;color:{CHARCOAL};margin-bottom:6px">
          ✍️  Win-back Subject Lines
          <span class="agency-badge">Agency</span>
        </div>
        <div style="font-size:11px;color:{GRAY_DRK};margin-bottom:12px">
          AI-generated email subject lines<br>
          for your win-back campaign.
        </div>
      </div>
    """, unsafe_allow_html=True)

    if agency:
        if st.button("Generate Subject Lines", key="subj_btn", use_container_width=True):
            if not ai_ready:
                st.info("Enter your API key in the sidebar.", icon="🤖")
            else:
                at_risk_row = summary[summary["segment"] == "At Risk"]
                avg_days    = float(rfm[rfm["segment"].isin(["At Risk","Cannot Lose"])]["days_since"].mean() or 0)
                avg_spend   = float(at_risk_row["avg_spend"].iloc[0]) if len(at_risk_row) > 0 else 0
                count_ar    = int(at_risk_row["count"].iloc[0]) if len(at_risk_row) > 0 else 0

                with st.spinner("Writing subject lines…"):
                    try:
                        router = LLMRouter(
                            provider=selected_provider,
                            api_key=api_key,
                            ollama_url=st.session_state.get("ollama_url","http://localhost:11434"),
                        )
                        sys_p, usr_p = winback_subject_lines_prompt(
                            "At Risk", avg_days, avg_spend, count_ar
                        )
                        subj_text = router.complete(sys_p, usr_p, max_tokens=200)
                        st.session_state["subj_lines"] = subj_text
                    except LLMError as e:
                        st.error(f"AI error: {e}")

        if "subj_lines" in st.session_state:
            with st.expander("📬  Subject Lines", expanded=True):
                st.markdown(st.session_state["subj_lines"])
    else:
        st.markdown(
            f'<div style="font-size:11px;color:{GRAY_DRK};padding:4px 0">'
            f'Upgrade to Agency plan to unlock AI copywriting.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
  <div style="text-align:center;padding:16px;font-size:11px;color:{GRAY_DRK};
              border-top:1px solid {GRAY_LT}">
    KI DataLab · RFM Customer Segmentation Tool ·
    <a href="https://kidatalab.com" style="color:{GREEN1}">kidatalab.com</a>
  </div>
""", unsafe_allow_html=True)
