import json
import os
import re
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

try:
    import keyring
except ImportError:  # pragma: no cover - optional dependency
    keyring = None

try:
    from py_clob_client.client import ClobClient
except ImportError:  # pragma: no cover - optional dependency
    ClobClient = None


GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
EVENT_PAGE_SIZE = 500

# CLOB /prices-history enforces a maximum window size per request (varies by params).
PRICE_HISTORY_MAX_DAYS_PER_CALL = 30

PRICE_HISTORY_FIDELITY_SECONDS = 60

TARGET_SLUG_PREFIXES = ("fed-decision-in-", "fed-interest-rates-")
CENTRAL_TZ = ZoneInfo("America/Chicago")

POLYMARKET_OUTPUT_COLUMNS = [
    "event_slug",
    "event_title",
    "market_id",  # Gamma market id (kept for reference)
    "question",
    "decision_date",
    "rate_move_bps",
    "rate_move_label",
    "observed_ts",
    "observed_day",
    "close_price",
]


def _safe_get(url, params=None, timeout=20):
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Request failed for {url}: {exc}")
        return None


def _parse_iso_datetime(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _datetime_to_epoch_seconds(dt):
    return int(dt.timestamp()) if dt else None


def _parse_rate_move(question):
    text = (question or "").lower()

    if "no change" in text:
        return 0, "0bps"

    cut = re.search(r"(\d+)\s*\+?\s*bps?.*?(decrease|cut|lower)", text)
    if cut:
        bps = int(cut.group(1))
        if "+" in cut.group(0):
            return -bps, f"-{bps}+bps"
        return -bps, f"-{bps}bps"

    hike = re.search(r"(increase|hike|raise).*?(\d+)\s*\+?\s*bps?", text)
    if hike:
        bps = int(hike.group(2))
        if "+" in hike.group(0):
            return bps, f"+{bps}+bps"
        return bps, f"+{bps}bps"

    return None, None


def _extract_price_points(payload):
    if payload is None:
        return []

    points = []
    if isinstance(payload, list):
        points = payload
    elif isinstance(payload, dict):
        for key in ("history", "pricesHistory", "prices_history", "data", "results", "prices"):
            value = payload.get(key)
            if isinstance(value, list):
                points = value
                break

    normalized = []
    for point in points:
        if not isinstance(point, dict):
            continue
        ts = (
            point.get("t")
            or point.get("timestamp")
            or point.get("time")
            or point.get("startTs")
            or point.get("ts")
        )
        price = point.get("p") or point.get("price") or point.get("value") or point.get("close")
        if ts is None or price is None:
            continue
        dt = _parse_iso_datetime(ts)
        if dt is None and isinstance(ts, (int, float, str)):
            try:
                dt = datetime.fromtimestamp(int(float(ts)), tz=timezone.utc)
            except ValueError:
                continue
        if dt is None:
            continue
        normalized.append(
            {
                "observed_ts": int(dt.timestamp()),
                "observed_day": dt.date().isoformat(),
                "close_price": price,
            }
        )
    return normalized


def _load_private_key():
    env_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if env_key:
        return env_key
    if keyring is not None:
        keyring_username = os.getenv("POLYMARKET_KEYRING_USERNAME") or os.getenv("USER")
        if not keyring_username:
            return None
        try:
            return keyring.get_password("Polymarket", keyring_username)
        except Exception:
            return None
    return None


def _build_clob_headers():
    private_key = _load_private_key()
    if not private_key or ClobClient is None:
        return {}
    try:
        client = ClobClient(host=CLOB_BASE, chain_id=137, key=private_key)
        creds = client.create_or_derive_api_creds()
        headers = {
            "POLY_API_KEY": creds.api_key,
            "POLY_PASSPHRASE": creds.api_passphrase,
            "POLY_SIGNATURE": creds.api_secret,
        }
        return {k: v for k, v in headers.items() if v}
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        print(f"Unable to initialize CLOB credentials: {exc}")
        return {}


def _first_clob_token_id(market):
    """
    Gamma market objects include clobTokenIds as a JSON-encoded string list, e.g.
      "[\"<YES_TOKEN>\", \"<NO_TOKEN>\"]"
    We want the first token id to query CLOB /prices-history.
    """
    if not isinstance(market, dict):
        return None

    raw = market.get("clobTokenIds") or market.get("clobTokenId")
    if raw is None:
        return None

    if isinstance(raw, list):
        return str(raw[0]) if raw else None

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
            if isinstance(parsed, str) and parsed:
                return str(parsed)
        except json.JSONDecodeError:
            # fallback if it's already a token id string
            return s

    if isinstance(raw, (int, float)):
        return str(int(raw))

    return None


def fetch_all_fed_events():
    """
    Fetch only events whose slug starts with TARGET_SLUG_PREFIXES.
    (This is stricter than checking for 'fed' in the title/slug.)
    """
    events = []
    offset = 0

    while True:
        page = _safe_get(
            f"{GAMMA_BASE}/events",
            params={"limit": EVENT_PAGE_SIZE, "offset": offset},
        )
        if not isinstance(page, list) or not page:
            break

        for event in page:
            slug = str(event.get("slug", "")).lower()
            if slug.startswith(TARGET_SLUG_PREFIXES):
                events.append(event)

        if len(page) < EVENT_PAGE_SIZE:
            break
        offset += EVENT_PAGE_SIZE

    return events


def _fetch_event_markets(event):
    markets = event.get("markets")
    if isinstance(markets, list) and markets:
        return markets

    slug = event.get("slug")
    if not slug:
        return []

    payload = _safe_get(f"{GAMMA_BASE}/events/slug/{slug}")
    if isinstance(payload, dict) and isinstance(payload.get("markets"), list):
        return payload["markets"]
    return []


def _fetch_market_history(clob_token_id, start_ts, end_ts, headers):
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())

    if not clob_token_id or not start_ts:
        return []

    start_ts = int(start_ts)

    # If no end date (open market), fetch up to now
    if not end_ts:
        end_ts = now_ts
    end_ts = int(min(end_ts, now_ts))

    if end_ts <= start_ts:
        return []

    all_points = []
    cur_start = start_ts

    # Start moderately; shrink if server rejects
    chunk_days = 1

    while cur_start < end_ts:
        chunk_end = min(cur_start + chunk_days * 86400, end_ts)

        params = {
            "market": str(clob_token_id),
            "startTs": int(cur_start),
            "endTs": int(chunk_end),
            "fidelity": PRICE_HISTORY_FIDELITY_SECONDS,
        }

        resp = requests.get(
            f"{CLOB_BASE}/prices-history",
            params=params,
            headers=headers,
            timeout=20,
        )

        #if resp.status_code == 400 and "interval is too long" in (resp.text or ""):
            #chunk_days = max(1, chunk_days // 2)
            # At 1 day chunks we still proceed; we should not bail out entirely.
            #if chunk_days == 1:
                # retry with 1-day windows; do not return []
            #    continue

        if resp.status_code in (401, 403):
            print(f"Skipping token {clob_token_id}: auth unavailable")
            return []

        resp.raise_for_status()
        all_points.extend(_extract_price_points(resp.json()))
        cur_start = chunk_end

    deduped = {p["observed_ts"]: p for p in all_points}
    return [deduped[k] for k in sorted(deduped)]


def _filter_to_2pm_central(df):
    if df.empty:
        return df

    working = df.copy()
    observed_utc = pd.to_datetime(working["observed_ts"], unit="s", utc=True, errors="coerce")
    working = working.loc[observed_utc.notna()].copy()
    if working.empty:
        return working

    observed_utc = observed_utc.loc[working.index]
    observed_central = observed_utc.dt.tz_convert(CENTRAL_TZ)
    working["_observed_day_central"] = observed_central.dt.date.astype(str)
    working["_observed_time_central"] = observed_central.dt.time

    eligible = working.loc[working["_observed_time_central"] <= time(14, 5)].copy()
    if eligible.empty:
        return eligible.drop(columns=["_observed_day_central", "_observed_time_central"])

    idx = eligible.groupby(
        ["event_slug", "market_id", "_observed_day_central"], dropna=False
    )["observed_ts"].idxmax()
    filtered = eligible.loc[idx].copy()
    filtered["observed_day"] = filtered["_observed_day_central"]
    return (
        filtered.drop(columns=["_observed_day_central", "_observed_time_central"])
        .sort_values(["event_slug", "market_id", "observed_ts"])
        .reset_index(drop=True)
    )


def build_polymarket_rates_dataframe():
    events = fetch_all_fed_events()
    headers = _build_clob_headers()
    rows = []

    for event in events:
        event_slug = event.get("slug")
        event_title = event.get("title")
        markets = _fetch_event_markets(event)

        for market in markets:
            gamma_market_id = market.get("id")
            question = market.get("question") or market.get("title")
            rate_move_bps, rate_move_label = _parse_rate_move(question)

            clob_token_id = _first_clob_token_id(market)

            start_dt = _parse_iso_datetime(
                market.get("createdAt") or market.get("startDate") or event.get("startDate")
            )
            end_dt = _parse_iso_datetime(
                market.get("closedAt") or market.get("endDate") or event.get("endDate")
            )
            decision_date = end_dt.date().isoformat() if end_dt else None
            start_ts = _datetime_to_epoch_seconds(start_dt)
            end_ts = _datetime_to_epoch_seconds(end_dt)

            history_rows = []
            if clob_token_id is not None and start_ts:
                history_rows = _fetch_market_history(clob_token_id, start_ts, end_ts, headers=headers)

            if not history_rows:
                # Keep writing these for now; we drop them at the end (but we want slugs list)
                rows.append(
                    {
                        "event_slug": event_slug,
                        "event_title": event_title,
                        "market_id": gamma_market_id,
                        "question": question,
                        "decision_date": decision_date,
                        "rate_move_bps": rate_move_bps,
                        "rate_move_label": rate_move_label,
                        "observed_ts": None,
                        "observed_day": None,
                        "close_price": None,
                    }
                )
                continue

            for point in history_rows:
                rows.append(
                    {
                        "event_slug": event_slug,
                        "event_title": event_title,
                        "market_id": gamma_market_id,
                        "question": question,
                        "decision_date": decision_date,
                        "rate_move_bps": rate_move_bps,
                        "rate_move_label": rate_move_label,
                        "observed_ts": point["observed_ts"],
                        "observed_day": point["observed_day"],
                        "close_price": point["close_price"],
                    }
                )

    df = pd.DataFrame.from_records(rows, columns=POLYMARKET_OUTPUT_COLUMNS)

    # Identify unmatched (no prices)
    unmatched = df[df["close_price"].isna()]
    if not unmatched.empty:
        # state the slugs (with counts)
        slug_counts = (
            unmatched.groupby("event_slug", dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        print("\nUnmatched slugs (no prices):")
        for slug, cnt in slug_counts.items():
            print(f"  {slug}: {cnt} rows")

    # Drop unmatched rows
    matched_df = df.dropna(subset=["close_price"]).reset_index(drop=True)
    return _filter_to_2pm_central(matched_df)


def main():
    df = build_polymarket_rates_dataframe()
    out_path = os.path.join("Data", "Polymarket", "Polymarket_rates.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
