import requests
from datetime import datetime, time as dt_time
import pandas as pd
import time
from zoneinfo import ZoneInfo


BASE = "https://api.elections.kalshi.com/trade-api/v2"
WINDOW_TRIM_SECONDS = 1
WINDOW_TRIM_BOUNDARY_MULTIPLIER = 2
MAX_TS_SCALE_STEPS = 4  # ns -> us -> ms -> s
# Pull candles in 1-day windows to avoid large-range rate limits.
KALSHI_CANDLE_CHUNK_SECONDS = 24 * 60 * 60
KALSHI_REQUEST_SLEEP_SECONDS = 0.2
KALSHI_MAX_RETRY_ATTEMPTS = 3
KALSHI_RETRY_BACKOFF_SECONDS = 1.0
CENTRAL_TZ = ZoneInfo("America/Chicago")

def fetch_all_historical_markets():
    """
    Fetch markets from Kalshi, paging until no more results.
    Returns a list of market dicts.
    """
    all_markets = []
    next_cursor = None
    params = {"limit": 1000, "series_ticker": "KXFEDDECISION"}

    while True:
        if next_cursor:
            params["cursor"] = next_cursor
        try:
            resp = requests.get(f"{BASE}/markets", params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Kalshi request error: {e}")
            break

        try:
            data = resp.json()
        except ValueError:
            print("Kalshi returned non-JSON response")
            break

        # try common locations for list of markets
        page_items = data.get("data") or data.get("markets") or data.get("results") or []
        if isinstance(page_items, dict):
            page_items = page_items.get("data") or page_items.get("results") or []

        if not page_items:
            break

        all_markets.extend(page_items)

        # detect cursor for next page
        next_cursor = (
            data.get("cursor")
            or data.get("next_cursor")
            or (data.get("meta") and data["meta"].get("cursor"))
            or (data.get("meta") and data["meta"].get("next_cursor"))
        )

        # stop if no cursor or fewer items than page size
        if not next_cursor or len(page_items) < params["limit"]:
            break

    return all_markets

def group_by_decision(markets):
    grouped = {}
    for m in markets:
        # e.g. 'KXFEDDECISION-26JAN-C25' → 'KXFEDDECISION-26JAN'
        parts = m["ticker"].rsplit("-", 1)
        key = parts[0]

        grouped.setdefault(key, []).append(m["ticker"])
    return grouped

def fetch_daily_candles(ticker, start_ts, end_ts):
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": 60,
    }
    url = f"{BASE}/historical/markets/{ticker}/candlesticks"
    resp = requests.get(url, params=params)

    if resp.status_code == 404:
        print(f"Not found (maybe not archived): {ticker}")
        return None

    return resp.json()

def _normalize_epoch_scale(v):
    for _ in range(MAX_TS_SCALE_STEPS):
        if v <= 10_000_000_000:
            return v
        v //= 1000
    return None

def _to_epoch_seconds(val):
    """Normalize possible timestamp formats to UNIX seconds (int)."""
    if val is None:
        return None
    # numeric (seconds or ms)
    if isinstance(val, (int, float)):
        v = int(val)
        # heuristics: handle ms/us/ns values
        v = _normalize_epoch_scale(v)
        return v
    # string -> try ISO format
    if isinstance(val, str):
        s = val.strip()
        # handle trailing 'Z'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            return int(dt.timestamp())
        except Exception:
            # fallback: try parse as int string
            try:
                iv = int(s)
                iv = _normalize_epoch_scale(iv)
                return iv
            except Exception:
                return None
    return None

def _series_from_ticker(ticker):
    """Extract series ticker from Kalshi market ticker."""
    if not isinstance(ticker, str) or not ticker:
        return None
    return ticker.split("-", 1)[0]

def _nested_value(d, key, nested_key):
    return (d.get(key) or {}).get(nested_key)

def _extract_candle_items(candles_json):
    if isinstance(candles_json, dict):
        items = (
            candles_json.get("candlesticks")
            or candles_json.get("data")
            or candles_json.get("results")
            or candles_json.get("candles")
            or candles_json.get("items")
        )
        if items is None:
            for value in candles_json.values():
                if isinstance(value, list):
                    items = value
                    break
        return items or []
    if isinstance(candles_json, list):
        return candles_json
    return []

def _request_with_retry(url, params, timeout):
    for attempt in range(KALSHI_MAX_RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == KALSHI_MAX_RETRY_ATTEMPTS - 1:
                raise exc
            time.sleep(KALSHI_RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue

        if resp.status_code == 429 and attempt < KALSHI_MAX_RETRY_ATTEMPTS - 1:
            time.sleep(KALSHI_RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue
        return resp

def fetch_market_times(ticker):
    """
    Fetch market metadata for ticker and return (created_time_sec, close_time_sec)
    Both returned as UNIX epoch seconds (int) or (None, None) on error.
    """
    url = f"{BASE}/markets/{ticker}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Kalshi market fetch error for {ticker}: {e}")
        return None, None

    try:
        data = resp.json()
    except ValueError:
        print(f"Non-JSON response when fetching market {ticker}")
        return None, None

    # common locations for timestamps
    created = (
        data.get("created_time")
        or (data.get("market") and data["market"].get("created_time"))
        or (data.get("data") and data["data"].get("created_time"))
    )
    close = (
        data.get("close_time")
        or (data.get("market") and data["market"].get("close_time"))
        or (data.get("data") and data["data"].get("close_time"))
    )

    created_ts = _to_epoch_seconds(created)
    close_ts = _to_epoch_seconds(close)

    if created_ts is None or close_ts is None:
        # try other common keys
        if isinstance(data, dict):
            # search shallowly
            for k in ("created_at", "created", "open_time", "start_time"):
                if created_ts is None and k in data:
                    created_ts = _to_epoch_seconds(data[k])
                if close_ts is None and k in data:
                    close_ts = _to_epoch_seconds(data[k])
    return created_ts, close_ts

def fetch_candles_using_market_times(ticker, period_interval=60):
    """
    Fetch candlesticks for a market using its created_time and close_time.
    Returns the parsed JSON from the candlesticks endpoint or None.
    """
    created_ts, close_ts = fetch_market_times(ticker)
    if not created_ts or not close_ts:
        print(f"Could not determine times for {ticker}: created={created_ts} close={close_ts}")
        return None

    start_ts = created_ts
    series_ticker = _series_from_ticker(ticker)
    if not series_ticker:
        print(f"Invalid ticker for candle fetch: {ticker}")
        return None
    urls = [
        f"{BASE}/series/{series_ticker}/markets/{ticker}/candlesticks",
        f"{BASE}/historical/markets/{ticker}/candlesticks",
    ]

    all_items = []
    cur_start = start_ts
    while cur_start < close_ts:
        chunk_end = min(cur_start + KALSHI_CANDLE_CHUNK_SECONDS, close_ts)
        chunk_params = {
            "start_ts": cur_start,
            "end_ts": chunk_end,
            "period_interval": period_interval,
            "include_latest_before_start": True,
        }

        resp = None
        for url in urls:
            try:
                resp = _request_with_retry(url, chunk_params, timeout=15)
                if resp is None:
                    continue
                if resp.status_code == 404:
                    continue
                if resp.status_code == 400 and chunk_end - cur_start > (WINDOW_TRIM_BOUNDARY_MULTIPLIER * WINDOW_TRIM_SECONDS):
                    narrowed_params = dict(chunk_params)
                    narrowed_params["start_ts"] = cur_start + WINDOW_TRIM_SECONDS
                    narrowed_params["end_ts"] = chunk_end - WINDOW_TRIM_SECONDS
                    resp = _request_with_retry(url, narrowed_params, timeout=15)
                    if resp is None:
                        continue
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"Error fetching candlesticks for {ticker} via {url}: {e}")
                resp = None
                continue

        if resp is None:
            print(f"Candles not found for {ticker} [{cur_start}, {chunk_end}]")
            return None

        try:
            all_items.extend(_extract_candle_items(resp.json()))
        except ValueError:
            print(f"Non-JSON candlestick response for {ticker}")
            return None

        cur_start = chunk_end
        if cur_start < close_ts:
            time.sleep(KALSHI_REQUEST_SLEEP_SECONDS)

    return {"candlesticks": all_items}

def _filter_to_2pm_central(df):
    if df.empty or "candle_ts" not in df.columns:
        return df

    working = df.copy()
    observed_utc = pd.to_datetime(working["candle_ts"], unit="s", utc=True, errors="coerce")
    working = working.loc[observed_utc.notna()].copy()
    if working.empty:
        return working

    observed_utc = observed_utc.loc[working.index]
    observed_central = observed_utc.dt.tz_convert(CENTRAL_TZ)
    working["_obs_day_central"] = observed_central.dt.date.astype(str)
    working["_obs_time_central"] = observed_central.dt.time

    # Keep the latest print on each Central calendar day at or before 2:00pm CT.
    eligible = working.loc[working["_obs_time_central"] <= dt_time(14, 5)].copy()
    if eligible.empty:
        return eligible.drop(columns=["_obs_day_central", "_obs_time_central"])

    group_cols = [col for col in ("decision_key", "ticker", "_obs_day_central") if col in eligible.columns]
    idx = eligible.groupby(group_cols, dropna=False)["candle_ts"].idxmax()
    filtered = eligible.loc[idx].copy()
    return (
        filtered.drop(columns=["_obs_day_central", "_obs_time_central"])
        .sort_values([col for col in ("decision_key", "ticker", "candle_ts") if col in filtered.columns])
        .reset_index(drop=True)
    )

def fetch_all_decision_candles():
    """
    Iterate all pulled markets (grouped by decision), fetch market times and candlesticks
    for each ticker, and return a single pandas DataFrame with all candle rows.
    """
    markets = fetch_all_historical_markets()
    grouped = group_by_decision(markets)

    records = []
    for decision_key, tickers in grouped.items():
        for ticker in tickers:
            created_ts, close_ts = fetch_market_times(ticker)
            if created_ts is None or close_ts is None:
                print(f"Skipping {ticker}: missing times")
                continue

            candles_json = fetch_candles_using_market_times(ticker)
            if not candles_json:
                continue

            items = _extract_candle_items(candles_json)

            if not items:
                print(f"No candle items for {ticker}")
                continue

            for c in items:
                # detect Kalshi candlestick structure (example provided by user)
                if isinstance(c, dict) and ("end_period_ts" in c or "price" in c):
                    ts = c.get("end_period_ts") or c.get("end_ts") or c.get("period_ts")
                    ts = _to_epoch_seconds(ts) if ts is not None else None

                    price = c.get("price") or {}
                    open_p = price.get("open") or price.get("open")
                    high_p = price.get("high") or price.get("max")
                    low_p = price.get("low") or price.get("min")
                    close_p = price.get("close") or price.get("previous")

                    # volume lives at top level in example
                    vol = c.get("volume") or c.get("v")
                else:
                    # fallback: normalize common candle fields
                    ts = c.get("time") or c.get("timestamp") or c.get("t") or c.get("start_ts")
                    ts = _to_epoch_seconds(ts) if ts is not None else None
                    open_p = c.get("open_dollars") or c.get("o") or c.get("start") or c.get("open_price")
                    high_p = c.get("high_dollars") or c.get("h")
                    low_p = c.get("low_dollars") or c.get("l")
                    close_p = c.get("close_dollars") or c.get("c") or c.get("end") or c.get("close_price")
                    vol = c.get("volume") or c.get("v")

                records.append({
                    "decision_key": decision_key,
                    "ticker": ticker,
                    "market_created_ts": created_ts,
                    "market_close_ts": close_ts,
                    "candle_ts": ts,
                    "open_price": open_p,
                    "high_price": high_p,
                    "low_price": low_p,
                    "close_price": close_p,
                    "mean_price": _nested_value(c, "price", "mean_dollars"),
                    "yes_bid_close": _nested_value(c, "yes_bid", "close_dollars"),
                    "yes_ask_close": _nested_value(c, "yes_ask", "close_dollars"),
                    "volume": vol,
                })

    df = pd.DataFrame.from_records(records)
    return _filter_to_2pm_central(df)


def main():

    # Pull all decisions and tickers and append into a single DataFrame
    print("Fetching candlesticks for all markets...")
    df = fetch_all_decision_candles()

    if df is None or df.empty:
        print("No candle data retrieved.")
    else:
        print("Combined candles DataFrame shape:", df.shape)
        print(df.head())
        try:
            import os
            out_path = os.path.join("Data", "Kalshi", "Kalshi_rates.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"Saved combined DataFrame to {out_path}")
        except Exception as e:
            print(f"Failed to save DataFrame to CSV: {e}")


if __name__ == "__main__":
    main()
