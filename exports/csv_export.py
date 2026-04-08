"""
exports/csv_export.py
---------------------
All CSV export functions. Each returns (bytes, filename) ready for
st.download_button(data=..., file_name=...).
"""

import pandas as pd
import io
from datetime import datetime
from calculations.segments import get_customers_by_segment, get_high_value_at_risk


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue().encode("utf-8")

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def export_full_rfm(rfm_df: pd.DataFrame) -> tuple[bytes, str]:
    cols = ["customer_id","customer_name","last_order_date","days_since",
            "order_count","total_spend","avg_order_value",
            "r_score","f_score","m_score","rfm_total","segment"]
    export_cols = [c for c in cols if c in rfm_df.columns]
    df = rfm_df[export_cols].copy()
    if "last_order_date" in df.columns:
        df["last_order_date"] = pd.to_datetime(df["last_order_date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("rfm_total", ascending=False).reset_index(drop=True)
    return _to_csv_bytes(df), f"rfm_full_export_{_timestamp()}.csv"


def export_segment(rfm_df: pd.DataFrame, segment_name: str) -> tuple[bytes, str]:
    df = get_customers_by_segment(rfm_df, segment_name)
    cols = ["customer_id","customer_name","last_order_date","days_since",
            "order_count","total_spend","r_score","f_score","m_score","rfm_total"]
    export_cols = [c for c in cols if c in df.columns]
    df = df[export_cols].copy()
    if "last_order_date" in df.columns:
        df["last_order_date"] = pd.to_datetime(df["last_order_date"]).dt.strftime("%Y-%m-%d")
    slug = segment_name.lower().replace(" ", "_")
    return _to_csv_bytes(df), f"rfm_{slug}_{_timestamp()}.csv"


def export_vip_list(rfm_df: pd.DataFrame) -> tuple[bytes, str]:
    frames = [get_customers_by_segment(rfm_df, s) for s in ["Champions","Loyal Customers"]]
    if not frames or all(len(f) == 0 for f in frames):
        df = pd.DataFrame(columns=["customer_id","customer_name","total_spend","order_count","segment"])
    else:
        df = pd.concat(frames, ignore_index=True)
    cols = ["customer_id","customer_name","total_spend","order_count","segment"]
    export_cols = [c for c in cols if c in df.columns]
    df = df[export_cols].sort_values("total_spend", ascending=False).reset_index(drop=True)
    return _to_csv_bytes(df), f"vip_list_meta_ads_{_timestamp()}.csv"


def export_churn_risk(rfm_df: pd.DataFrame, min_spend: float = 0.0) -> tuple[bytes, str]:
    df = get_high_value_at_risk(rfm_df, min_spend=min_spend)
    cols = ["customer_id","customer_name","segment","last_order_date","days_since",
            "order_count","total_spend","r_score","f_score","m_score"]
    export_cols = [c for c in cols if c in df.columns]
    df = df[export_cols].copy()
    if "last_order_date" in df.columns:
        df["last_order_date"] = pd.to_datetime(df["last_order_date"]).dt.strftime("%Y-%m-%d")
    return _to_csv_bytes(df), f"churn_risk_{_timestamp()}.csv"


def export_winback_campaign(rfm_df: pd.DataFrame) -> tuple[bytes, str]:
    frames = [get_customers_by_segment(rfm_df, s) for s in ["At Risk","Cannot Lose","About To Sleep"]]
    if not frames or all(len(f) == 0 for f in frames):
        df = pd.DataFrame(columns=["customer_id","customer_name","segment","total_spend","days_since"])
    else:
        df = pd.concat(frames, ignore_index=True)
    cols = ["customer_id","customer_name","segment","last_order_date","days_since",
            "order_count","total_spend","r_score","f_score","m_score"]
    export_cols = [c for c in cols if c in df.columns]
    df = df[export_cols].sort_values(["segment","total_spend"], ascending=[True,False]).reset_index(drop=True)
    if "last_order_date" in df.columns:
        df["last_order_date"] = pd.to_datetime(df["last_order_date"]).dt.strftime("%Y-%m-%d")
    return _to_csv_bytes(df), f"winback_campaign_{_timestamp()}.csv"


def export_klaviyo(rfm_df: pd.DataFrame, segment_name: str) -> tuple[bytes, str]:
    df = get_customers_by_segment(rfm_df, segment_name).copy()
    slug = segment_name.lower().replace(" ", "_")
    if df.empty:
        out = pd.DataFrame(columns=["Email","First Name","Last Name","RFM Segment","Total Spend","Order Count","Days Since Last Purchase","RFM Score","Customer ID"])
        return _to_csv_bytes(out), f"klaviyo_{slug}_{_timestamp()}.csv"
    name_parts = df["customer_name"].str.strip().str.split(" ", n=1, expand=True)
    df["First Name"] = name_parts[0].fillna("")
    df["Last Name"]  = name_parts[1].fillna("") if 1 in name_parts.columns else ""
    is_email = df["customer_id"].str.contains("@", na=False)
    df["Email"] = df["customer_id"].where(is_email, other="")
    out = pd.DataFrame({"Email": df["Email"], "First Name": df["First Name"], "Last Name": df["Last Name"],
                        "RFM Segment": segment_name, "Total Spend": df["total_spend"].round(2),
                        "Order Count": df["order_count"], "Days Since Last Purchase": df["days_since"],
                        "RFM Score": df["rfm_total"], "Customer ID": df["customer_id"]})
    return _to_csv_bytes(out), f"klaviyo_{slug}_{_timestamp()}.csv"


def export_mailchimp(rfm_df: pd.DataFrame, segment_name: str) -> tuple[bytes, str]:
    df = get_customers_by_segment(rfm_df, segment_name).copy()
    slug = segment_name.lower().replace(" ", "_")
    if df.empty:
        out = pd.DataFrame(columns=["Email Address","First Name","Last Name","TAGS"])
        return _to_csv_bytes(out), f"mailchimp_{slug}_{_timestamp()}.csv"
    name_parts = df["customer_name"].str.strip().str.split(" ", n=1, expand=True)
    df["First Name"] = name_parts[0].fillna("")
    df["Last Name"]  = name_parts[1].fillna("") if 1 in name_parts.columns else ""
    is_email = df["customer_id"].str.contains("@", na=False)
    df["Email Address"] = df["customer_id"].where(is_email, other="")
    out = pd.DataFrame({"Email Address": df["Email Address"], "First Name": df["First Name"],
                        "Last Name": df["Last Name"], "TAGS": segment_name})
    return _to_csv_bytes(out), f"mailchimp_{slug}_{_timestamp()}.csv"
