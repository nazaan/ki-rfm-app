"""
utils/data_loader.py
--------------------
Handles CSV ingestion, validation, and normalisation.
Returns a clean DataFrame with guaranteed column names:
    customer_id | customer_name | order_date | order_value

All downstream modules can assume these exact column names exist.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Tuple


# ── Column name aliases ────────────────────────────────────────────────────
# Maps common export formats (Shopify, Etsy, WooCommerce, generic) to
# our internal names. Add more aliases here as needed.

CUSTOMER_ID_ALIASES = [
    "customer_id", "customerid", "customer id", "client_id", "clientid",
    "user_id", "userid", "buyer_id", "buyerid", "contact_id",
    "email",  # acceptable as a unique ID if no numeric ID present
    "customer email", "customer_email",
]

CUSTOMER_NAME_ALIASES = [
    "customer_name", "customername", "customer name", "client_name",
    "name", "full_name", "fullname", "billing_name", "buyer_name",
    "shipping_name", "contact_name",
]

ORDER_DATE_ALIASES = [
    "order_date", "orderdate", "order date", "date", "transaction_date",
    "transactiondate", "purchase_date", "created_at", "createdat",
    "order_created", "paid_at", "processed_at", "sale_date",
]

ORDER_VALUE_ALIASES = [
    "order_value", "ordervalue", "order value", "amount", "total",
    "order_total", "ordertotal", "revenue", "price", "subtotal",
    "sale_amount", "transaction_amount", "lineitem_price",
    "total_price", "grand_total", "net_revenue",
]


# ── Main entry point ───────────────────────────────────────────────────────

def load_and_validate(file) -> Tuple[pd.DataFrame, list]:
    """
    Load a CSV file uploaded via Streamlit's file_uploader.
    Returns (clean_df, warnings) where warnings is a list of non-fatal
    issues the UI can display to the user.

    Raises ValueError with a user-friendly message on fatal errors.
    """
    warnings = []

    # 1. Read raw CSV
    raw_df = _read_csv(file)

    # 2. Normalise column names (lowercase, strip whitespace)
    raw_df.columns = [c.strip().lower().replace(" ", "_") for c in raw_df.columns]

    # 3. Map columns to internal names
    df, col_warnings = _map_columns(raw_df)
    warnings.extend(col_warnings)

    # 4. Clean each column
    df, clean_warnings = _clean_columns(df)
    warnings.extend(clean_warnings)

    # 5. Drop rows that are completely unusable
    df, drop_warnings = _drop_bad_rows(df)
    warnings.extend(drop_warnings)

    # 6. Final shape check
    if len(df) < 2:
        raise ValueError(
            "After cleaning, fewer than 2 valid rows remain. "
            "Please check your CSV format and try again."
        )

    return df, warnings


# ── Step 1: Read CSV ───────────────────────────────────────────────────────

def _read_csv(file) -> pd.DataFrame:
    """Try common encodings and separators."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    separators = [",", ";", "\t", "|"]

    for enc in encodings:
        for sep in separators:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, sep=sep,
                                 on_bad_lines="skip", low_memory=False)
                if len(df.columns) >= 2 and len(df) >= 1:
                    return df
            except Exception:
                continue

    raise ValueError(
        "Could not read this file. Please make sure it is a valid CSV "
        "with comma, semicolon, or tab separators."
    )


# ── Step 2: Map columns ────────────────────────────────────────────────────

def _map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Attempt to match raw column names to internal names.
    Returns renamed DataFrame and list of warnings.
    """
    warnings = []
    cols = list(df.columns)
    mapping = {}

    def find_col(aliases: list, label: str, required: bool) -> str | None:
        for alias in aliases:
            normalised = alias.lower().replace(" ", "_")
            if normalised in cols:
                return normalised
        # fuzzy fallback: partial match
        for alias in aliases:
            for col in cols:
                if alias.replace("_", "") in col.replace("_", ""):
                    warnings.append(
                        f"Column '{col}' matched to '{label}' — "
                        f"please verify this is correct."
                    )
                    return col
        if required:
            raise ValueError(
                f"Could not find a '{label}' column. "
                f"Expected one of: {', '.join(aliases[:5])}. "
                f"Your columns are: {', '.join(cols)}."
            )
        return None

    cid   = find_col(CUSTOMER_ID_ALIASES,   "Customer ID",   required=True)
    cname = find_col(CUSTOMER_NAME_ALIASES, "Customer Name", required=False)
    odate = find_col(ORDER_DATE_ALIASES,    "Order Date",    required=True)
    oval  = find_col(ORDER_VALUE_ALIASES,   "Order Value",   required=True)

    if cname is None:
        warnings.append(
            "No 'Customer Name' column found — names will be left blank."
        )

    # Build renamed DataFrame with only the columns we need
    rename_map = {cid: "customer_id", odate: "order_date", oval: "order_value"}
    keep_cols  = [cid, odate, oval]

    if cname:
        rename_map[cname] = "customer_name"
        keep_cols.append(cname)

    df = df[keep_cols].rename(columns=rename_map)

    if "customer_name" not in df.columns:
        df["customer_name"] = ""

    return df, warnings


# ── Step 3: Clean columns ──────────────────────────────────────────────────

def _clean_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    warnings = []

    # customer_id — stringify, strip whitespace
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["customer_id"] = df["customer_id"].replace("nan", np.nan)

    # customer_name — stringify, strip
    df["customer_name"] = df["customer_name"].astype(str).str.strip()
    df["customer_name"] = df["customer_name"].replace({"nan": "", "None": ""})

    # order_date — parse flexibly
    df["order_date"], date_warnings = _parse_dates(df["order_date"])
    warnings.extend(date_warnings)

    # order_value — strip currency symbols, parse as float
    df["order_value"], value_warnings = _parse_values(df["order_value"])
    warnings.extend(value_warnings)

    return df, warnings


def _parse_dates(series: pd.Series) -> Tuple[pd.Series, list]:
    """Try multiple date formats explicitly. Return parsed series + warnings."""
    warnings = []
    original_count = series.notna().sum()

    # Try these formats in order — most common first
    formats = [
        "%Y-%m-%d", "%Y/%m/%d",           # ISO
        "%d/%m/%Y", "%d-%m-%Y",           # European day-first
        "%m/%d/%Y", "%m-%d-%Y",           # US month-first
        "%d %b %Y", "%d %B %Y",           # 15 Nov 2024
        "%b %d %Y", "%B %d %Y",           # Nov 15 2024
        "%Y-%m-%d %H:%M:%S",              # with time
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]

    best = None
    best_count = 0

    for fmt in formats:
        try:
            attempt = pd.to_datetime(series, format=fmt, errors="coerce")
            valid = attempt.notna().sum()
            if valid > best_count:
                best_count = valid
                best = attempt
        except Exception:
            continue

    # Final fallback: let pandas infer
    if best is None or best_count == 0:
        best = pd.to_datetime(series, errors="coerce", dayfirst=True)
        best_count = best.notna().sum()

    parsed = best

    # Strip timezone if present
    if hasattr(parsed.dt, "tz") and parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)

    failed = original_count - best_count
    if failed > 0 and best_count > 0:
        warnings.append(
            f"{failed} date values could not be parsed and will be excluded."
        )
    elif best_count == 0:
        raise ValueError(
            "Could not parse any dates in your date column. "
            "Please use YYYY-MM-DD, DD/MM/YYYY, or MM/DD/YYYY format."
        )

    return parsed, warnings


def _parse_values(series: pd.Series) -> Tuple[pd.Series, list]:
    """Strip currency symbols and thousands separators, parse as float."""
    warnings = []

    # Remove currency symbols, spaces, and thousands separators
    cleaned = (
        series.astype(str)
        .str.replace(r"[£$€¥₹,\s]", "", regex=True)
        .str.replace(r"\((.+)\)", r"-\1", regex=True)  # (100) → -100
        .str.strip()
    )

    parsed = pd.to_numeric(cleaned, errors="coerce")

    negative_count = (parsed < 0).sum()
    if negative_count > 0:
        warnings.append(
            f"{negative_count} negative order values found (refunds?). "
            f"These will be excluded from scoring."
        )

    failed = parsed.isna().sum()
    if failed > 0:
        warnings.append(
            f"{failed} order values could not be parsed and will be excluded."
        )

    return parsed, warnings


# ── Step 4: Drop bad rows ──────────────────────────────────────────────────

def _drop_bad_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    warnings = []
    original_len = len(df)

    # Drop rows missing critical fields
    df = df.dropna(subset=["customer_id", "order_date", "order_value"])

    # Drop zero or negative order values
    df = df[df["order_value"] > 0]

    # Drop obviously bad customer IDs
    df = df[df["customer_id"].str.len() > 0]
    df = df[~df["customer_id"].isin(["nan", "none", "null", "n/a", "-"])]

    dropped = original_len - len(df)
    if dropped > 0:
        warnings.append(
            f"{dropped} rows removed due to missing or invalid data."
        )

    return df.reset_index(drop=True), warnings


# ── Utility: summary for UI ────────────────────────────────────────────────

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Returns a quick summary dict for display in the UI after upload.
    """
    return {
        "total_transactions": len(df),
        "unique_customers":   df["customer_id"].nunique(),
        "date_range_start":   df["order_date"].min().strftime("%d %b %Y"),
        "date_range_end":     df["order_date"].max().strftime("%d %b %Y"),
        "total_revenue":      df["order_value"].sum(),
        "avg_order_value":    df["order_value"].mean(),
    }
