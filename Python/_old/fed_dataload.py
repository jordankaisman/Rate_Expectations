def add_fed_decisions(df, rows):
    """
    Append FED decision rows to a DataFrame.

    rows: list of dicts with keys:
      - decision_date: 'YYYY-MM'
      - decision: 'hold' | 'cut' | 'hike'
      - decision_bp: int or str (e.g. -25, 0, +25)
      - notes: str
    Returns: new concatenated DataFrame
    """

    VALID = {"hold", "cut", "hike"}
    validated = []
    for i, r in enumerate(rows):
        if not all(k in r for k in ("decision_date", "decision", "decision_bp", "notes")):
            raise ValueError(f"Row {i} missing required keys")
        # validate date format YYYY-MM
        try:
            datetime.strptime(r["decision_date"], "%Y-%m")
        except Exception:
            raise ValueError(f"Invalid decision_date format in row {i}: {r['decision_date']}")
        # validate decision category
        dec = str(r["decision"]).lower()
        if dec not in VALID:
            raise ValueError(f"Invalid decision in row {i}: {r['decision']}")
        # validate/normalize decision_bp to integer (bp)
        bp = r["decision_bp"]
        try:
            if isinstance(bp, str):
                bp_int = int(bp.replace("+", ""))
            else:
                bp_int = int(bp)
        except Exception:
            raise ValueError(f"Invalid decision_bp in row {i}: {r['decision_bp']}")
        validated.append({
            "decision_date": r["decision_date"],
            "decision": dec,
            "decision_bp": bp_int,
            "notes": r["notes"],
        })

    new = pd.DataFrame(validated)
    return pd.concat([df, new], ignore_index=True)


#Old

# Kalshi # kxfeddecision-26apr

import requests

url = "https://api.elections.kalshi.com/trade-api/v2/historical/markets/kxfeddecision-26jan/candlesticks"

response = requests.get(url)

print(response.text)

# Get series information for kxfeddecision
url = "https://api.elections.kalshi.com/trade-api/v2/series/kxfeddecision"
response = requests.get(url)
series_data = response.json()

print("series_data type:", type(series_data))
print("top-level keys:", list(series_data.keys()))

if "series" in series_data and isinstance(series_data["series"], dict):
    print("series keys:", list(series_data["series"].keys()))
    print("title:", series_data["series"].get("title"))
    print("frequency:", series_data["series"].get("frequency"))
    print("category:", series_data["series"].get("category"))


print(f"Series Title: {series_data['series']['title']}")
print(f"Frequency: {series_data['series']['frequency']}")
print(f"Category: {series_data['series']['category']}")

# Get all open markets for the kxfeddecision series
markets_url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=kxfeddecision&status=open"
markets_response = requests.get(markets_url)
markets_data = markets_response.json()

print(f"\nActive markets in kxfeddecision series:")
for market in markets_data['markets']:
    print(f"- {market['ticker']}: {market['title']}")
    print(f"  Event: {market['event_ticker']}")
    print(f"  Yes Price: {market['yes_price']}¢ | Volume: {market['volume']}")
    print()

# Get details for a specific event if you have its ticker
if markets_data['markets']:
    # Let's get details for the first market's event
    event_ticker = markets_data['markets'][0]['event_ticker']
    event_url = f"https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}"
    event_response = requests.get(event_url)
    event_data = event_response.json()

    print(f"Event Details:")
    print(f"Title: {event_data['event']['title']}")
    print(f"Category: {event_data['event']['category']}")