import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical rate-move category mapping tables
# ---------------------------------------------------------------------------

# Kalshi: variant_token = raw ticker suffix (last '-' component).
# Required remaps per spec: H26/C26 -> H25+/C25+; C24 -> C50+ (via C26).
KALSHI_VARIANT_TO_CANONICAL = {
    "C24":  "C50+",   # C24 -> C26 -> C50+ (net remap)
    "C25":  "C25",
    "C26":  "C50+",
    "C>25": "C50+",
    "H0":   "H0",
    "H25":  "H25",
    "H26":  "H50+",
    "H50":  "H50",
    "H>25": "H50+",
    # T-prefixed variants (legacy "target" contracts)
    "TC25": "C25",
    "TH50": "H50",
}

# Polymarket: variant_token = compact descriptor derived from question text
# (e.g. "decreases 25", "no change", "increases 25+").
# Canonical labels follow C/H prefix convention.
POLYMARKET_VARIANT_TO_CANONICAL = {
    # No change / hold
    "no change":    "H0",
    "increases 0":  "H0",
    # Hikes
    "increases 25":  "H25",
    "increases 25+": "H25+",
    "increases 50":  "H50",
    "increases 50+": "H50+",
    "increases 75":  "H75",
    "increases 75+": "H75+",
    # Cuts
    "decreases 25":  "C25",
    "decreases 50":  "C50",
    "decreases 50+": "C50+",
    "decreases 75":  "C75",
    "decreases 75+": "C75+",
}

# Also map existing rate_move_label strings from the CSV to canonical.
POLYMARKET_LABEL_TO_CANONICAL = {
    "0bps":    "H0",
    "+0bps":   "H0",
    "+25bps":  "H25",
    "+25+bps": "H25+",
    "+50bps":  "H50",
    "+50+bps": "H50+",
    "+75bps":  "H75",
    "+75+bps": "H75+",
    "-25bps":  "C25",
    "-50bps":  "C50",
    "-50+bps": "C50+",
    "-75bps":  "C75",
    "-75+bps": "C75+",
}


# ---------------------------------------------------------------------------
# Helper: derive Polymarket variant_token from question text
# ---------------------------------------------------------------------------

def _poly_variant_token(question: str) -> str | None:
    """
    Return a compact descriptor like 'decreases 25', 'increases 25+', or
    'no change' derived from the question text.  Returns None if unknown.
    """
    text = (question or "").lower()

    # "No change" check first (some questions contain "raise" but lead with
    # "No change" or "increase...0 bps").
    if "no change" in text:
        return "no change"
    m0 = re.search(r"(increase|raise)\w*\s+interest\s+rates\s+by\s+0\s+bps", text)
    if m0:
        return "increases 0"

    # Cuts / decreases
    if re.search(r"(decrease|cut|lower)", text):
        m = re.search(r"(\d+)\s*(\+)?\s*bps", text)
        if m:
            bps = m.group(1)
            plus = "+" if m.group(2) else ""
            return f"decreases {bps}{plus}"

    # Hikes / increases / raises
    if re.search(r"(increase|raise|hike)", text):
        m = re.search(r"(\d+)\s*(\+)?\s*bps", text)
        if m:
            bps = m.group(1)
            plus = "+" if m.group(2) else ""
            return f"increases {bps}{plus}"

    return None


# ---------------------------------------------------------------------------
# Decision-date mapping: Kalshi -> Polymarket  (YYMMM key)
# ---------------------------------------------------------------------------

def _build_yymmm_mapping(polymarket_df: pd.DataFrame) -> dict[str, str]:
    """
    Build a mapping  {YYMMM_key: decision_date}  from Polymarket data.

    The YYMMM key is strftime('%y%b').upper() applied to decision_date,
    e.g. '2025-06-18' -> '25JUN'.  This matches the format used in Kalshi
    decision_key suffixes (e.g. KXFEDDECISION-25JUN -> key '25JUN').

    If multiple decision_dates map to the same key, a warning is printed and
    the earliest date is chosen.
    """
    mapping: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}

    dates = polymarket_df["decision_date"].dropna().unique()
    for d in dates:
        try:
            dt = pd.to_datetime(d)
        except Exception:
            continue
        key = dt.strftime("%y%b").upper()
        if key in mapping:
            conflicts.setdefault(key, {mapping[key]})
            conflicts[key].add(d)
            # Choose earliest
            if d < mapping[key]:
                mapping[key] = d
        else:
            mapping[key] = d

    if conflicts:
        print("\n[WARNING] Multiple Polymarket decision_dates map to the same YYMMM key:")
        for k, dates_set in conflicts.items():
            print(f"  {k}: {sorted(dates_set)} -> using earliest '{mapping[k]}'")

    return mapping


def _kalshi_yymmm_key(decision_key: str) -> str | None:
    """
    Extract the 5-char YYMMM key from a Kalshi decision_key.

    Examples:
      'KXFEDDECISION-25JUN'  -> '25JUN'
      'FEDDECISION-24JAN31'  -> '24JAN'   (strip trailing day digits)
    """
    if not isinstance(decision_key, str):
        return None
    parts = decision_key.split("-")
    if len(parts) < 2:
        return None
    # The meeting identifier is everything after the first dash segment.
    meeting = parts[-1] if len(parts) == 2 else "-".join(parts[1:])
    # For compound keys like '24JAN31', keep only the first 5 chars (YYMMM).
    return meeting[:5] if len(meeting) >= 5 else None


# ---------------------------------------------------------------------------
# Build long-format DataFrames for each venue
# ---------------------------------------------------------------------------

def _build_kalshi_long(path: str, yymmm_mapping: dict[str, str]) -> pd.DataFrame:
    kalshi = pd.read_csv(path)

    # --- 1. Variant token & canonical label ---
    kalshi["variant_token"] = kalshi["ticker"].str.rsplit("-", n=1).str[-1]
    all_tokens = kalshi["variant_token"].dropna().unique()
    mapped_tokens = {t for t in all_tokens if t in KALSHI_VARIANT_TO_CANONICAL}
    unmapped_tokens = sorted(set(all_tokens) - mapped_tokens)
    print("\n[Kalshi] Variant tokens found:", sorted(all_tokens))
    print("[Kalshi] Unmapped variant tokens:", unmapped_tokens)

    kalshi["canonical_label"] = kalshi["variant_token"].map(KALSHI_VARIANT_TO_CANONICAL)

    # --- 2. Decision date from Polymarket YYMMM mapping ---
    kalshi["yymmm_key"] = kalshi["decision_key"].map(_kalshi_yymmm_key)
    kalshi["decision_date"] = kalshi["yymmm_key"].map(yymmm_mapping)

    mapped_count = kalshi["decision_date"].notna().sum()
    missing_count = kalshi["decision_date"].isna().sum()
    print(f"\n[Kalshi] decision_date mapped: {mapped_count} rows, missing: {missing_count} rows")
    missing_keys = sorted(kalshi.loc[kalshi["decision_date"].isna(), "yymmm_key"].dropna().unique())
    print(f"[Kalshi] Missing YYMMM keys (no Polymarket match): {missing_keys}")

    # --- 3. Price ---
    bid = pd.to_numeric(kalshi.get("yes_bid_close"), errors="coerce")
    ask = pd.to_numeric(kalshi.get("yes_ask_close"), errors="coerce")
    #close = pd.to_numeric(kalshi.get("price_close_dollars"), errors="coerce")
    #fallback_close = pd.to_numeric(kalshi.get("close"), errors="coerce")
    midpoint = (bid + ask) / 2.0
    midpoint = midpoint.where(bid.notna() & ask.notna())
    kalshi["price"] = midpoint

    # --- 4. Observed day in PST (America/Los_Angeles) ---
    kalshi["observed_ts"] = pd.to_numeric(kalshi.get("candle_ts"), errors="coerce")
    ts_utc = pd.to_datetime(kalshi["observed_ts"], unit="s", utc=True, errors="coerce")
    kalshi["observed_day_pst"] = ts_utc.dt.tz_convert("America/Los_Angeles").dt.date.astype(str)
    # Keep original UTC day for diagnostic comparison
    kalshi["observed_day_utc"] = ts_utc.dt.date.astype(str)

    # Diagnostic sample
    sample_cols = ["observed_ts", "observed_day_utc", "observed_day_pst"]
    print("\n[Kalshi] Sample observed_ts / observed_day comparison (first 5 rows with valid ts):")
    sample = kalshi.loc[kalshi["observed_ts"].notna(), sample_cols].head(5)
    print(sample.to_string(index=False))

    out = kalshi[["decision_date", "observed_day_pst", "observed_ts", "canonical_label", "price"]].copy()
    out = out.dropna(subset=["decision_date", "observed_day_pst", "observed_ts", "canonical_label", "price"])
    out["venue"] = "kalshi"
    return out


def _build_polymarket_long(path: str, yymmm_mapping: dict[str, str]) -> pd.DataFrame:
    try:
        poly = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        poly = pd.DataFrame(
            columns=[
                "decision_date", "observed_ts", "close_price",
                "rate_move_label", "question",
            ]
        )

    poly["observed_ts"] = pd.to_numeric(poly.get("observed_ts"), errors="coerce")
    poly["close_price"] = pd.to_numeric(poly.get("close_price"), errors="coerce")

    # --- 1. Observed day in PST (America/Los_Angeles) ---
    ts_utc = pd.to_datetime(poly["observed_ts"], unit="s", utc=True, errors="coerce")
    poly["observed_day_pst"] = ts_utc.dt.tz_convert("America/Los_Angeles").dt.date.astype(str)
    poly["observed_day_utc"] = ts_utc.dt.date.astype(str)

    # Diagnostic sample
    existing_day_col = "observed_day" if "observed_day" in poly.columns else "observed_day_utc"
    sample_cols = ["observed_ts", existing_day_col, "observed_day_pst"]
    print("\n[Polymarket] Sample observed_ts / observed_day comparison (first 5 rows with valid ts):")
    sample = poly.loc[poly["observed_ts"].notna(), sample_cols].head(5)
    print(sample.to_string(index=False))

    # --- 2. Canonical label from question text (primary) or rate_move_label (fallback) ---
    poly["variant_token"] = poly.get("question", pd.Series(dtype=str)).map(_poly_variant_token)
    poly["canonical_label"] = poly["variant_token"].map(POLYMARKET_VARIANT_TO_CANONICAL)

    # Fallback: map existing rate_move_label to canonical
    fallback_mask = poly["canonical_label"].isna() & poly.get("rate_move_label", pd.Series(dtype=str)).notna()
    if fallback_mask.any():
        poly.loc[fallback_mask, "canonical_label"] = (
            poly.loc[fallback_mask, "rate_move_label"].map(POLYMARKET_LABEL_TO_CANONICAL)
        )

    all_vtokens = poly["variant_token"].dropna().unique()
    unmapped_vtokens = sorted(
        {t for t in all_vtokens if POLYMARKET_VARIANT_TO_CANONICAL.get(t) is None}
    )
    print("\n[Polymarket] Variant tokens found:", sorted(all_vtokens))
    print("[Polymarket] Unmapped variant tokens:", unmapped_vtokens)

    # --- 3. Ensure decision_date is present (comes from the CSV; verify it matches mapping) ---
    # decision_date in Polymarket CSV is the source of truth - no re-mapping needed.

    out = poly[["decision_date", "observed_day_pst", "observed_ts", "canonical_label", "close_price"]].copy()
    out = out.rename(columns={"close_price": "price"})
    out = out.dropna(subset=["decision_date", "observed_day_pst", "observed_ts", "canonical_label", "price"])
    out["venue"] = "polymarket"
    return out


# ---------------------------------------------------------------------------
# Daily last observation + wide pivot
# ---------------------------------------------------------------------------

def _last_observation_per_day(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the last observation (by observed_ts) for each:
      (decision_date, observed_day_pst, canonical_label, venue)
    """
    if long_df.empty:
        return long_df

    long_df = long_df.sort_values(
        ["decision_date", "observed_day_pst", "canonical_label", "venue", "observed_ts"]
    )
    last = long_df.groupby(
        ["decision_date", "observed_day_pst", "canonical_label", "venue"], as_index=False
    ).tail(1)

    return last[["decision_date", "observed_day_pst", "canonical_label", "venue", "price"]]


def _pivot_wide(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per (decision_date, observed_day_pst), wide columns:
      kalshi_<canonical_label>, polymarket_<canonical_label>
    """
    if daily_df.empty:
        return pd.DataFrame(columns=["decision_date", "observed_day_pst"])

    daily_df = daily_df.copy()
    daily_df["col"] = daily_df["venue"] + "_" + daily_df["canonical_label"].astype(str)

    wide = daily_df.pivot_table(
        index=["decision_date", "observed_day_pst"],
        columns="col",
        values="price",
        aggfunc="first",
    ).reset_index()

    wide.columns = [str(c) for c in wide.columns]
    return wide


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def merge_kalshi_polymarket(kalshi_path: str, polymarket_path: str) -> pd.DataFrame:
    # Load Polymarket first to build the YYMMM -> decision_date mapping.
    try:
        poly_raw = pd.read_csv(polymarket_path)
    except pd.errors.EmptyDataError:
        poly_raw = pd.DataFrame(columns=["decision_date"])

    yymmm_mapping = _build_yymmm_mapping(poly_raw)
    print(f"\n[Mapping] Built {len(yymmm_mapping)} YYMMM -> decision_date entries")

    kalshi_long = _build_kalshi_long(kalshi_path, yymmm_mapping)
    poly_long = _build_polymarket_long(polymarket_path, yymmm_mapping)

    print(f"\n[Long] Kalshi rows after filtering: {len(kalshi_long)}")
    print(f"[Long] Polymarket rows after filtering: {len(poly_long)}")

    both = pd.concat([kalshi_long, poly_long], ignore_index=True)

    daily = _last_observation_per_day(both)
    wide = _pivot_wide(daily)

    # Each row is a unique (decision_date, observed_day_pst) combination.
    return wide.sort_values(["decision_date", "observed_day_pst"])


def main():
    kalshi_path = os.path.join("Data", "Kalshi", "Kalshi_rates.csv")
    polymarket_path = os.path.join("Data", "Polymarket", "Polymarket_rates.csv")
    out_path = os.path.join("Data", "Merged", "Prediction_all.csv")

    merged = merge_kalshi_polymarket(kalshi_path, polymarket_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nMerged rows: {len(merged)}")
    print(f"Merged columns: {list(merged.columns)}")
    print(f"Saved merged output to {out_path}")


if __name__ == "__main__":
    main()