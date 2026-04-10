from __future__ import annotations

from datetime import date
from typing import Any, Iterable

def discount_factor_from_nodes(
    target_date: date,
    node_dates: list[date],
    node_dfs: list[float],
    valuation_date: date,
) -> float:
    if target_date == valuation_date:
        return 1.0
    if not node_dates:
        raise ValueError("node_dates cannot be empty")
    if len(node_dates) != len(node_dfs):
        raise ValueError("node_dates and node_dfs must have the same length")

    if target_date <= node_dates[0]:
        return node_dfs[0]

    target_t = year_fraction_act360(valuation_date, target_date)
    times = [year_fraction_act360(valuation_date, d) for d in node_dates]

    for i in range(1, len(node_dates)):
        if target_date <= node_dates[i]:
            t0, t1 = times[i - 1], times[i]
            d0, d1 = node_dfs[i - 1], node_dfs[i]
            if abs(t1 - t0) < 1e-15:
                return d1
            w = (target_t - t0) / (t1 - t0)
            return math.exp(math.log(d0) + (math.log(d1) - math.log(d0)) * w)

    if len(node_dates) == 1:
        return node_dfs[0]
    t0, t1 = times[-2], times[-1]
    d0, d1 = node_dfs[-2], node_dfs[-1]
    if abs(t1 - t0) < 1e-15:
        return d1
    slope = (math.log(d1) - math.log(d0)) / (t1 - t0)
    return math.exp(math.log(d1) + slope * (target_t - t1))


def bootstrap_ois_discount_curve(
    valuation_date: date, instruments: list[dict], insert_coupon_nodes: bool = True
) -> tuple[list[date], list[float]]:
    ordered = sorted(instruments, key=lambda x: x["maturity_date"])
    node_dates: list[date] = [valuation_date]
    node_dfs: list[float] = [1.0]
    dfs_by_date: dict[date, float] = {valuation_date: 1.0}
    eps = 1e-12

    for inst in ordered:
        maturity_date = inst["maturity_date"]
        if maturity_date <= valuation_date or maturity_date in dfs_by_date:
            continue
        rate = float(inst["rate"])
        kind = inst.get("kind", "ois_swap")

        if kind == "deposit":
            accrual = year_fraction_act360(valuation_date, maturity_date)
            df_t = 1.0 / (1.0 + rate * accrual)
        else:
            maturity_months = int(inst.get("maturity_months") or months_between(valuation_date, maturity_date))
            pay_dates, accruals = fixed_leg_schedule(valuation_date, maturity_date, maturity_months)
            a_known = 0.0
            for i, pay_date in enumerate(pay_dates[:-1]):
                df_i = dfs_by_date.get(pay_date)
                if df_i is None:
                    df_i = discount_factor_from_nodes(pay_date, node_dates, node_dfs, valuation_date)
                a_known += accruals[i] * df_i
            alpha_n = accruals[-1]
            df_t = (1.0 - rate * a_known) / (1.0 + rate * alpha_n)

        prev_df = node_dfs[-1]
        if df_t > prev_df:
            warnings.warn(
                f"Non-monotone DF at {maturity_date.isoformat()} ({df_t:.12f} > {prev_df:.12f}); clamping",
                RuntimeWarning,
            )
            clamped = max(prev_df - eps, prev_df * (1.0 - 1e-9))
            df_t = max(clamped, eps)
            if df_t >= prev_df:
                df_t = prev_df * (1.0 - 1e-9)
        if df_t <= 0:
            raise ValueError(f"Invalid discount factor {df_t} at {maturity_date.isoformat()}")

        dfs_by_date[maturity_date] = df_t
        node_dates.append(maturity_date)
        node_dfs.append(df_t)

        if insert_coupon_nodes and kind == "ois_swap":
            maturity_months_val = int(
                inst.get("maturity_months") or months_between(valuation_date, maturity_date)
            )
            coupon_pay_dates, _ = fixed_leg_schedule(valuation_date, maturity_date, maturity_months_val)
            for pay_date in coupon_pay_dates[:-1]:
                if pay_date <= valuation_date or pay_date in dfs_by_date:
                    continue
                coupon_df = discount_factor_from_nodes(pay_date, node_dates, node_dfs, valuation_date)
                idx = bisect.bisect_left(node_dates, pay_date)
                node_dates.insert(idx, pay_date)
                node_dfs.insert(idx, coupon_df)
                dfs_by_date[pay_date] = coupon_df

    return node_dates, node_dfs


def check_curve_parity(
    valuation_date: date,
    instruments: list[dict],
    node_dates: list[date],
    node_dfs: list[float],
) -> tuple[list[dict], float]:
    """Compute implied rates from the bootstrapped curve and compare to quoted rates.

    Returns a list of per-instrument parity results and the maximum absolute error.
    """
    results: list[dict] = []
    max_error = 0.0
    for inst in sorted(instruments, key=lambda x: x["maturity_date"]):
        maturity_date = inst["maturity_date"]
        if maturity_date <= valuation_date:
            continue
        rate = float(inst["rate"])
        kind = inst.get("kind", "ois_swap")

        df_mat = discount_factor_from_nodes(maturity_date, node_dates, node_dfs, valuation_date)

        if kind == "deposit":
            accrual = year_fraction_act360(valuation_date, maturity_date)
            implied_rate = (1.0 / df_mat - 1.0) / max(accrual, 1e-15)
        else:
            maturity_months = int(
                inst.get("maturity_months") or months_between(valuation_date, maturity_date)
            )
            pay_dates, accruals = fixed_leg_schedule(valuation_date, maturity_date, maturity_months)
            annuity = sum(
                a * discount_factor_from_nodes(d, node_dates, node_dfs, valuation_date)
                for a, d in zip(accruals, pay_dates)
            )
            implied_rate = (1.0 - df_mat) / max(annuity, 1e-15)

        error = implied_rate - rate
        abs_error = abs(error)
        if abs_error > max_error:
            max_error = abs_error

        results.append(
            {
                "maturity_date": maturity_date.isoformat(),
                "kind": kind,
                "quoted_rate": rate,
                "implied_rate": implied_rate,
                "error": error,
            }
        )
    return results, max_error

class OISCurve:
    def __init__(
        self,
        valuation_date: date,
        node_dates: list[date],
        node_dfs: list[float],
        interpolation: str,
        meeting_dates: Iterable[date | str] | None = None,
    ):
        if len(node_dates) != len(node_dfs):
            raise ValueError("node_dates and discount_factors must have same length")
        if len(node_dates) < 2:
            raise ValueError("curve must have at least two nodes")
        self.valuation_date = valuation_date
        self.node_dates = node_dates
        self.times = [_year_fraction_act360(self.valuation_date, d) for d in node_dates]
        self.discount_factors = node_dfs
        self.interpolation = interpolation
        self.meeting_dates = sorted(
            _to_date(m) for m in (meeting_dates or []) if _to_date(m) > self.valuation_date
        )
        self._step_boundaries: list[float] | None = None
        self._step_forwards: list[float] | None = None

    def _discount_factor_loglinear_time(self, t: float) -> float:
        t = max(t, 0.0)
        if t <= self.times[0]:
            return self.discount_factors[0]
        if t >= self.times[-1]:
            t0, t1 = self.times[-2], self.times[-1]
            d0, d1 = self.discount_factors[-2], self.discount_factors[-1]
            slope = (math.log(d1) - math.log(d0)) / (t1 - t0)
            return math.exp(math.log(d1) + slope * (t - t1))

        for i in range(1, len(self.times)):
            if t <= self.times[i]:
                t0, t1 = self.times[i - 1], self.times[i]
                d0, d1 = self.discount_factors[i - 1], self.discount_factors[i]
                if self.interpolation == "linear_zero_rates":
                    z0 = -math.log(d0) / t0 if t0 > 0 else -math.log(d1) / t1
                    z1 = -math.log(d1) / t1
                    w = (t - t0) / (t1 - t0)
                    z = z0 + (z1 - z0) * w
                    return math.exp(-z * t)
                w = (t - t0) / (t1 - t0)
                ln_d = math.log(d0) + (math.log(d1) - math.log(d0)) * w
                return math.exp(ln_d)
        return self.discount_factors[-1]

    def _ensure_step_segments(self) -> None:
        if self._step_boundaries is not None and self._step_forwards is not None:
            return

        boundaries = [0.0]
        max_t = self.times[-1]
        for m in self.meeting_dates:
            t = _year_fraction_act360(self.valuation_date, m)
            if 0.0 < t < max_t:
                boundaries.append(t)
        boundaries.append(max_t)
        boundaries = sorted(set(boundaries))
        if len(boundaries) < 2:
            boundaries = [0.0, max_t]

        forwards: list[float] = []
        for i in range(1, len(boundaries)):
            t0 = boundaries[i - 1]
            t1 = boundaries[i]
            d0 = self._discount_factor_loglinear_time(t0)
            d1 = self._discount_factor_loglinear_time(t1)
            forwards.append(-math.log(d1 / d0) / max(t1 - t0, 1e-12))

        self._step_boundaries = boundaries
        self._step_forwards = forwards

    def discount_factor(self, target: date | float) -> float:
        t = _year_fraction_act360(self.valuation_date, target) if isinstance(target, date) else max(target, 0.0)
        if self.interpolation != "flat_forward_meetings":
            return self._discount_factor_loglinear_time(t)

        self._ensure_step_segments()
        boundaries = self._step_boundaries or [0.0, self.times[-1]]
        forwards = self._step_forwards or [self.instantaneous_forward(0.0)]
        if t <= 0.0:
            return 1.0

        if t >= boundaries[-1]:
            t0 = boundaries[-1]
            d0 = self._discount_factor_loglinear_time(t0)
            return d0 * math.exp(-forwards[-1] * (t - t0))

        for i in range(1, len(boundaries)):
            if t <= boundaries[i]:
                t0 = boundaries[i - 1]
                d0 = self._discount_factor_loglinear_time(t0)
                return d0 * math.exp(-forwards[i - 1] * (t - t0))
        return self._discount_factor_loglinear_time(t)

    def instantaneous_forward(self, target: date | float) -> float:
        t = _year_fraction_act360(self.valuation_date, target) if isinstance(target, date) else max(target, 0.0)
        if self.interpolation == "flat_forward_meetings":
            self._ensure_step_segments()
            boundaries = self._step_boundaries or [0.0, self.times[-1]]
            forwards = self._step_forwards or [0.0]
            if t <= 0.0:
                return forwards[0]
            if t >= boundaries[-1]:
                return forwards[-1]
            for i in range(1, len(boundaries)):
                if t <= boundaries[i]:
                    return forwards[i - 1]
            return forwards[-1]
        if t <= self.times[0]:
            t = self.times[0]
        for i in range(1, len(self.times)):
            if t <= self.times[i]:
                t0, t1 = self.times[i - 1], self.times[i]
                d0, d1 = self.discount_factors[i - 1], self.discount_factors[i]
                return -(math.log(d1) - math.log(d0)) / (t1 - t0)
        t0, t1 = self.times[-2], self.times[-1]
        d0, d1 = self.discount_factors[-2], self.discount_factors[-1]
        return -(math.log(d1) - math.log(d0)) / (t1 - t0)

    def average_rate(self, start: date, end: date) -> float:
        if end <= start:
            return 0.0
        t0 = _year_fraction_act360(self.valuation_date, start)
        t1 = _year_fraction_act360(self.valuation_date, end)
        d0 = self.discount_factor(t0)
        d1 = self.discount_factor(t1)
        return -math.log(d1 / d0) / max(t1 - t0, 1e-12)



def build_curve(market_data: dict[str, Any], meetings: Iterable[date | str] | None = None) -> OISCurve:
    valuation_date = _to_date(market_data["valuation_date"])
    overnight_rate = _normalize_rate(float(market_data["overnight_rate"]))
    ois_swaps = market_data.get("ois_swaps", [])
    instruments: list[dict[str, Any]] = []
    for row in ois_swaps:
        rate = _normalize_rate(float(row["rate"]))
        if "maturity_date" in row and row["maturity_date"]:
            maturity = _to_date(row["maturity_date"])
            maturity_months = months_between(valuation_date, maturity)
            kind = "deposit" if maturity_months <= 1 else "ois_swap"
        elif "tenor_months" in row:
            tenor_months = float(row["tenor_months"])
            if tenor_months < 0.2:
                maturity = add_days(valuation_date, 1)
                maturity_months = 0
                kind = "deposit"
            elif tenor_months < 1.0:
                weeks = max(int(round(tenor_months * APPROX_DAYS_PER_MONTH / DAYS_PER_WEEK)), 1)
                maturity = add_days(valuation_date, int(weeks * DAYS_PER_WEEK))
                maturity_months = 0
                kind = "deposit"
            else:
                maturity_months = int(round(tenor_months))
                maturity = add_months(valuation_date, maturity_months)
                kind = "ois_swap"
        elif "tenor_years" in row:
            maturity_months = int(round(float(row["tenor_years"]) * 12.0))
            maturity = add_months(valuation_date, maturity_months)
            kind = "ois_swap"
        else:
            continue
        if maturity <= valuation_date:
            continue
        instruments.append(
            {"maturity_date": maturity, "rate": rate, "kind": kind, "maturity_months": maturity_months}
        )

    if not any(inst["kind"] == "deposit" for inst in instruments):
        instruments.append(
            {
                "maturity_date": add_days(valuation_date, 1),
                "rate": overnight_rate,
                "kind": "deposit",
                "maturity_months": 0,
            }
        )

    try:
        node_dates, node_dfs = bootstrap_ois_discount_curve(valuation_date, instruments)
    except Exception as exc:
        import logging

        logging.warning(
            "OIS bootstrap failed for %s: %s; falling back to simple curve", valuation_date, exc
        )
        # Fallback: simple flat curve anchored at overnight rate
        node_dates = [valuation_date, add_days(valuation_date, 1)]
        t1 = _year_fraction_act360(valuation_date, node_dates[-1])
        node_dfs = [1.0, math.exp(-overnight_rate * t1)]
        seen_dates: set[date] = {valuation_date, node_dates[-1]}
        for inst in sorted(instruments, key=lambda x: x["maturity_date"]):
            mat = inst["maturity_date"]
            if mat <= valuation_date or mat in seen_dates:
                continue
            t = _year_fraction_act360(valuation_date, mat)
            node_dates.append(mat)
            node_dfs.append(math.exp(-overnight_rate * t))
            seen_dates.add(mat)

    curve = OISCurve(valuation_date, node_dates, node_dfs, CONFIG.interpolation, meeting_dates=meetings)

    if os.getenv("SOFR_BOOTSTRAP_DIAG") == "1":
        parity_results, max_err = check_curve_parity(valuation_date, instruments, node_dates, node_dfs)
        diag_dir = os.path.join("Data", "Diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        diag_path = os.path.join(diag_dir, f"sofr_parity_{valuation_date.isoformat()}.csv")
        try:
            import csv

            with open(diag_path, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["maturity_date", "kind", "quoted_rate", "implied_rate", "error"]
                )
                writer.writeheader()
                writer.writerows(parity_results)
        except Exception:
            import logging

            logging.warning("[SOFR_BOOTSTRAP_DIAG] max_parity_error=%.4e", max_err)
            for r in parity_results:
                logging.warning(
                    "[SOFR_BOOTSTRAP_DIAG] maturity=%s quoted=%.8f implied=%.8f error=%.4e",
                    r["maturity_date"],
                    r["quoted_rate"],
                    r["implied_rate"],
                    r["error"],
                )

    if os.getenv("SOFR_OIS_DIAGNOSTICS") == "1":
        for inst in sorted(instruments, key=lambda x: x["maturity_date"]):
            maturity = inst["maturity_date"]
            quote = inst["rate"]
            if inst["kind"] == "deposit":
                accrual = _year_fraction_act360(valuation_date, maturity)
                implied = (1.0 / curve.discount_factor(maturity) - 1.0) / max(accrual, 1e-12)
            else:
                pay_dates, accruals = fixed_leg_schedule(valuation_date, maturity, int(inst["maturity_months"]))
                annuity = sum(a * curve.discount_factor(d) for a, d in zip(accruals, pay_dates))
                implied = (1.0 - curve.discount_factor(maturity)) / max(annuity, 1e-12)
            print(
                f"[SOFR_OIS_DIAGNOSTICS] maturity={maturity.isoformat()} quote={quote:.8f} "
                f"implied={implied:.8f} error={implied - quote:.8e}"
            )

    return curve



def _delta_to_probabilities(delta: float, step_size: float) -> dict[str, float]:
    two_step = 2.0 * step_size
    d = max(min(delta, two_step), -two_step)

    probs = {
        "p_minus_50": 0.0,
        "p_minus_25": 0.0,
        "p_0": 0.0,
        "p_plus_25": 0.0,
        "p_plus_50": 0.0,
    }

    if 0.0 <= d <= step_size:
        p_plus_25 = d / step_size
        probs["p_plus_25"] = p_plus_25
        probs["p_0"] = 1.0 - p_plus_25
    elif step_size < d <= two_step:
        p_plus_50 = (d - step_size) / step_size
        probs["p_plus_50"] = p_plus_50
        probs["p_plus_25"] = 1.0 - p_plus_50
    elif -step_size <= d < 0.0:
        p_minus_25 = abs(d) / step_size
        probs["p_minus_25"] = p_minus_25
        probs["p_0"] = 1.0 - p_minus_25
    else:
        p_minus_50 = (abs(d) - step_size) / step_size
        probs["p_minus_50"] = p_minus_50
        probs["p_minus_25"] = 1.0 - p_minus_50

    for key in probs:
        probs[key] = min(1.0, max(0.0, probs[key]))

    total = sum(probs.values())
    if total <= 0:
        probs["p_0"] = 1.0
        total = 1.0

    for key in probs:
        probs[key] /= total
    return probs



def discretize_changes(expectations_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in expectations_table:
        probs = _delta_to_probabilities(float(row["expected_change"]), CONFIG.step_size)
        table.append(
            {
                "date": row["date"],
                "meeting_date": row["meeting_date"],
                **probs,
                "method": row["method"],
            }
        )
    return table



def apply_ois_anchor(
    monthly_table: list[dict[str, Any]],
    curve: OISCurve,
    meetings: Iterable[date | str],
    window_months: int,
) -> list[dict[str, Any]]:
    valuation_date = curve.valuation_date
    month_set: set[str] = set()

    for meeting in meetings:
        meeting_date = _to_date(meeting)
        month_diff = (meeting_date.year - valuation_date.year) * 12 + (meeting_date.month - valuation_date.month)
        if 0 <= month_diff <= window_months:
            month_set.add(_month_key(meeting_date))

    adjusted: list[dict[str, Any]] = []
    for row in monthly_table:
        new_row = dict(row)
        if row["month"] in month_set and row["source"] == "futures":
            m_date = datetime.strptime(f"{row['month']}-01", "%Y-%m-%d").date()
            new_row["expected_rate"] = curve.average_rate(m_date, _add_months(m_date, 1))
            new_row["source"] = "ois_anchor"
        adjusted.append(new_row)
    return adjusted




def _build_diagnostics(
    curve: OISCurve,
    monthly_table: list[dict[str, Any]],
    expectations: list[dict[str, Any]],
    probabilities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    discount_monotonic = all(
        curve.discount_factors[i] <= curve.discount_factors[i - 1] for i in range(1, len(curve.discount_factors))
    )

    expected_by_meeting = {row["meeting_date"]: row["expected_change"] for row in expectations}

    diagnostics: list[dict[str, Any]] = []
    for p in probabilities:
        weighted = (
            p["p_minus_50"] * -0.005
            + p["p_minus_25"] * -0.0025
            + p["p_plus_25"] * 0.0025
            + p["p_plus_50"] * 0.005
        )
        total_prob = p["p_minus_50"] + p["p_minus_25"] + p["p_0"] + p["p_plus_25"] + p["p_plus_50"]
        diagnostics.append(
            {
                "date": p["date"],
                "meeting_date": p["meeting_date"],
                "method": p["method"],
                "discount_monotonic": discount_monotonic,
                "probabilities_sum_to_1": abs(total_prob - 1.0) < 1e-8,
                "expected_change_consistent": abs(weighted - expected_by_meeting[p["meeting_date"]]) < 1e-8,
            }
        )

    ois_monthly: dict[str, float] = {}
    for row in monthly_table:
        if row["source"] == "ois":
            ois_monthly[row["month"]] = row["expected_rate"]

    for row in monthly_table:
        if row["source"] == "futures":
            ois_rate = ois_monthly.get(row["month"])
            if ois_rate is None:
                m_date = datetime.strptime(f"{row['month']}-01", "%Y-%m-%d").date()
                ois_rate = curve.average_rate(m_date, _add_months(m_date, 1))
            diff = abs(row["expected_rate"] - ois_rate)
            diagnostics.append(
                {
                    "date": row["date"],
                    "meeting_date": None,
                    "method": expectations[0]["method"] if expectations else "unknown",
                    "discount_monotonic": discount_monotonic,
                    "probabilities_sum_to_1": True,
                    "expected_change_consistent": True,
                    "futures_ois_divergence": diff,
                    "futures_ois_divergence_flag": diff > CONFIG.futures_ois_diff_threshold,
                }
            )

    return diagnostics



def run_variant(
    market_data: dict[str, Any],
    meetings: Iterable[date | str],
    window_months: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    curve = build_curve(market_data, meetings=meetings)
    monthly = build_monthly_expectations(curve, market_data.get("futures", []))

    if window_months is None:
        method = "raw_futures_ois"
        adjusted_monthly = monthly
    else:
        method_lookup = {1: "ois_anchor_short", 3: "ois_anchor_long"}
        method = method_lookup.get(window_months, f"ois_anchor_{window_months}m")
        adjusted_monthly = apply_ois_anchor(monthly, curve, meetings, window_months)

    expectations = map_meetings(adjusted_monthly, curve, meetings, method=method)
    probabilities = discretize_changes(expectations)
    diagnostics = _build_diagnostics(curve, adjusted_monthly, expectations, probabilities)
    return expectations, probabilities, diagnostics


def build_monthly_expectations(curve: OISCurve, futures: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    valuation_date = curve.valuation_date
    horizon_start = _month_start(valuation_date)
    futures_lookup: dict[str, float] = {}

    for row in futures:
        month = row.get("month")
        if not month and row.get("contract_month"):
            month = row["contract_month"]
        if not month:
            continue
        rate = 100.0 - float(row["price"])
        futures_lookup[month] = _normalize_rate(rate)

    ois_monthly: dict[str, float] = {}
    for i in range(CONFIG.forecast_months + 1):
        month_start = _add_months(horizon_start, i)
        month = _month_key(month_start)
        month_end = _add_months(month_start, 1)
        ois_monthly[month] = curve.average_rate(month_start, month_end)

    basis_by_month = {
        month: futures_lookup[month] - ois_monthly[month]
        for month in sorted(futures_lookup.keys())
        if month in ois_monthly
    }

    sorted_basis_months = sorted(basis_by_month.keys())
    last_basis = basis_by_month[sorted_basis_months[0]] if sorted_basis_months else 0.0
    table: list[dict[str, Any]] = []
    for i in range(CONFIG.forecast_months + 1):
        month_start = _add_months(horizon_start, i)
        month = _month_key(month_start)
        if month in basis_by_month:
            last_basis = basis_by_month[month]
        expected_rate = ois_monthly[month] + last_basis
        source = "ois_calibrated" if sorted_basis_months else "ois"
        if month not in basis_by_month and not sorted_basis_months:
            source = "ois"

        table.append(
            {
                "date": valuation_date.isoformat(),
                "month": month,
                "expected_rate": expected_rate,
                "source": source,
            }
        )
    return table


def map_meetings(
    monthly_table: list[dict[str, Any]],
    curve: OISCurve,
    meetings: Iterable[date | str],
    method: str = "raw_futures_ois",
) -> list[dict[str, Any]]:
    valuation_date = curve.valuation_date
    monthly_lookup = {row["month"]: row["expected_rate"] for row in monthly_table}

    horizon_end = valuation_date + timedelta(days=730)
    meeting_dates = sorted(
        d
        for d in (_to_date(m) for m in meetings)
        if CONFIG.meeting_start <= d <= CONFIG.meeting_end and valuation_date <= d <= horizon_end
    )

    table: list[dict[str, Any]] = []
    spot_rate = curve.average_rate(valuation_date, valuation_date + timedelta(days=1))
    for idx, meeting_date in enumerate(meeting_dates):
        effective_date = meeting_date + timedelta(days=1)
        prev_anchor = meeting_dates[idx - 1] if idx > 0 else valuation_date
        if prev_anchor >= effective_date:
            prev_anchor = effective_date - timedelta(days=1)

        if idx + 1 < len(meeting_dates):
            next_anchor = meeting_dates[idx + 1]
        else:
            next_anchor = _add_months(_month_start(meeting_date), 1)
        if next_anchor <= effective_date:
            next_anchor = _add_months(_month_start(effective_date), 1)

        if prev_anchor <= valuation_date < effective_date:
            spot_window_end = min(effective_date, valuation_date + timedelta(days=1))
            spot_days = max((spot_window_end - valuation_date).days, 0)
            forward_start = max(prev_anchor, spot_window_end)
        else:
            spot_days = 0
            forward_start = prev_anchor
        forward_pre_days = max((effective_date - forward_start).days, 0)
        if spot_days + forward_pre_days <= 0:
            pre_rate = 0.0
            forward_pre_rate = 0.0
        else:
            forward_pre_rate = _average_rate_from_monthly(monthly_lookup, curve, forward_start, effective_date)
            pre_rate = (
                spot_rate * spot_days + forward_pre_rate * forward_pre_days
            ) / max(spot_days + forward_pre_days, 1)

        post_rate = _average_rate_from_monthly(monthly_lookup, curve, effective_date, next_anchor)

        expected_change = post_rate - pre_rate

        table.append(
            {
                "date": valuation_date.isoformat(),
                "meeting_date": meeting_date.isoformat(),
                "effective_date": effective_date.isoformat(),
                "pre_rate": pre_rate,
                "post_rate": post_rate,
                "expected_change": expected_change,
                "spot_rate": spot_rate,
                "spot_days": spot_days,
                "forward_pre_rate": forward_pre_rate,
                "forward_pre_days": forward_pre_days,
                "method": method,
            }
        )

    return table


def run_legacy_bootstrap_robustness(
    market_data: dict[str, Any], meetings: Iterable[date | str]
) -> dict[str, list[dict[str, Any]]]:
    """
    Run legacy OIS bootstrap variants for robustness diagnostics only.

    The trade pipeline should consume trade_sr1_chain and trade_ois_forward_chain.
    This helper isolates the legacy variants (raw_futures_ois / ois_anchor_*) so
    they can be inspected separately without merging them into trade output.

    Returns the same structure as ``run_daily``:
      - expectations: meeting-level expected changes per legacy method
      - probabilities: discretized distributions per meeting/method
      - policy_path: implied post-meeting policy path
      - diagnostics: parity/consistency checks for legacy variants
    """
    from Python.data_engineering import sofr_expectations_pipeline as pipeline

    return pipeline.run_daily(market_data, meetings)


class SR1ChainEstimator:
    """
    Estimates FOMC meeting jumps using a recursive step-function logic 
    applied to 1-Month SOFR Futures (SR1).
    """
    def __init__(
        self,
        valuation_date: date,
        monthly_rates: dict[str, float],
        meetings: Iterable[date],
    ) -> None:
        self.valuation_date = valuation_date
        # monthly_rates keys are "YYYY-MM", values are (100 - Price)
        self.monthly_rates = {k: float(v) for k, v in monthly_rates.items()}
        # Filter for future meetings only
        self.meetings = sorted([m for m in meetings if m >= valuation_date])

    def estimate(self) -> list[dict[str, Any]]:
        if not self.monthly_rates:
            return []

        months = sorted(self.monthly_rates.keys())
        meeting_map = {f"{m.year:04d}-{m.month:02d}": m for m in self.meetings}

        # 1. Anchor Identification
        # We find the first month without a meeting to establish the baseline 'r_anchor'
        anchor_idx = next(
            (i for i, month in enumerate(months) if month not in meeting_map), 
            0
        )

        records: list[dict[str, Any]] = []
        prev_post: float | None = None
        # Weights format: { "YYYY-MM": weight_on_contract }
        prev_weights: dict[str, float] = {}

        # 2. Recursive Processing
        for idx in range(anchor_idx, len(months)):
            month_str = months[idx]
            year, month_num = map(int, month_str.split("-"))
            month_days = calendar.monthrange(year, month_num)[1]
            month_rate = self.monthly_rates[month_str]
            meeting_date = meeting_map.get(month_str)

            # Initialize anchor
            if prev_post is None:
                prev_post = month_rate
                prev_weights = {month_str: 1.0}
                continue

            r_pre = prev_post
            
            if meeting_date:
                # The jump becomes effective the day AFTER the meeting (Fed standard)
                # d = number of days at the OLD rate
                d = meeting_date.day 
                post_days = month_days - d
                
                # If a meeting is on the last day of the month, post_days = 0.
                # In this case, the jump does not affect THIS month's average (SR1_M).
                # We defer solving for the jump until the following month.
                if post_days > 0:
                    # Solve for r_post using weighted average:
                    # month_rate = (d/N)*r_pre + ((N-d)/N)*r_post
                    r_post = (month_days * month_rate - d * r_pre) / post_days
                    
                    # Recursive Linear Weights Update:
                    # w_new = alpha * Contract_Current + beta * w_prev
                    alpha = month_days / post_days  # Current month contract multiplier
                    beta = -d / post_days           # Previous rate multiplier
                    
                    weights = {k: beta * v for k, v in prev_weights.items()}
                    weights[month_str] = weights.get(month_str, 0.0) + alpha
                    
                    sensitivity = post_days / month_days
                else:
                    # Jump effective next month. Current month is purely the old rate.
                    r_post = r_pre
                    weights = dict(prev_weights)
                    sensitivity = 0.0
                
                jump = r_post - r_pre

                # Jump weights: vector delta between current and previous weights
                all_keys = set(weights) | set(prev_weights)
                jump_weights = {
                    k: weights.get(k, 0.0) - prev_weights.get(k, 0.0) 
                    for k in all_keys
                }

                records.append({
                    "meeting_date": meeting_date.isoformat(),
                    "effective_date": (meeting_date + timedelta(days=1)).isoformat(),
                    "month": month_str,
                    "r_pre": r_pre,
                    "r_post": r_post,
                    "jump_sr1": jump,
                    "sensitivity_sr1": sensitivity,
                    "recursive_weights_sr1": jump_weights,
                    "is_high_leverage": (0 < post_days < 5)
                })
            else:
                # No meeting: rate remains constant across this month
                r_post = r_pre
                jump = 0.0
                weights = dict(prev_weights)
            
            # Carry forward to next month
            prev_post = r_post
            prev_weights = weights

        return records
    
class OISForwardChain:
    """
    Estimates FOMC meeting jumps using a recursive step-function logic 
    across adjacent OIS nodes.
    """
    def __init__(
        self,
        valuation_date: date,
        ois_quotes: list[dict[str, float]],
        meetings: Iterable[date],
    ) -> None:
        self.valuation_date = valuation_date
        self.ois_quotes = ois_quotes
        # Only process meetings occurring on or after today
        self.meetings = sorted(m for m in meetings if m >= valuation_date)

    def _tenor_to_date(self, tenor_months: float) -> date:
        """Converts monthly tenors to maturity dates."""
        if tenor_months < 1.0:
            days = max(int(round(tenor_months * 30.0)), 1)
            return self.valuation_date + timedelta(days=days)
        
        # Standard add_months logic
        d = self.valuation_date
        y = d.year + (d.month - 1 + int(round(tenor_months))) // 12
        m = (d.month - 1 + int(round(tenor_months))) % 12 + 1
        last_day = calendar.monthrange(y, m)[1]
        return date(y, m, min(d.day, last_day))

    def estimate(self) -> list[dict[str, Any]]:
        if not self.ois_quotes:
            return []

        # 1. Parse OIS points and calculate year fractions (ACT/360)
        points: list[tuple[date, float, float]] = []
        for row in self.ois_quotes:
            tenor = float(row["tenor_months"])
            if tenor <= 0.0: continue
            maturity = self._tenor_to_date(tenor)
            # Simple ACT/360 year fraction
            t = (maturity - self.valuation_date).days / 360.0
            if t <= 0.0: continue
            points.append((maturity, t, float(row["rate"])))

        points = sorted(points, key=lambda x: x[0])
        if len(points) < 2:
            return []

        # 2. Identify the Anchor (first gap with no meeting)
        anchor_gap_idx = 1
        for i in range(1, len(points)):
            start, end = points[i - 1][0], points[i][0]
            # A meeting affects a gap if its EFFECTIVE date (meeting+1) falls within it
            if not any(start < (m + timedelta(days=1)) <= end for m in self.meetings):
                anchor_gap_idx = i
                break

        records: list[dict[str, Any]] = []
        prev_post: float | None = None
        prev_weights: dict[str, float] = {}

        # 3. Recursive Chain Logic
        for i in range(anchor_gap_idx, len(points)):
            start_date, t1, r1 = points[i - 1]
            end_date, t2, r2 = points[i]
            gap_days = (end_date - start_date).days
            if gap_days <= 0: continue

            # Discrete Forward Rate for the gap [T1, T2]
            forward = (r2 * t2 - r1 * t1) / (t2 - t1)
            gap_key = start_date.isoformat()

            # Find a meeting whose jump becomes effective inside this gap
            meeting = next(
                (m for m in self.meetings if start_date < (m + timedelta(days=1)) <= end_date),
                None
            )

            # Initialize anchor
            if prev_post is None:
                prev_post = forward
                prev_weights = {gap_key: 1.0}
                if meeting is None: continue

            r_pre = prev_post
            if meeting:
                eff_date = meeting + timedelta(days=1)
                pre_days = (eff_date - start_date).days
                post_days = (end_date - eff_date).days
                
                # SENSITIVITY CHECK: If post_days is 0, the meeting has no impact on this gap's average.
                # The jump logic is deferred to the NEXT gap.
                if post_days > 0:
                    # r_post calculation using weighted average of the gap
                    # Forward = (pre_days/gap_days)*r_pre + (post_days/gap_days)*r_post
                    r_post = (gap_days * forward - pre_days * r_pre) / post_days
                    
                    alpha = gap_days / post_days  # Current forward multiplier
                    beta = -pre_days / post_days  # Previous rate multiplier
                    
                    weights = {k: beta * v for k, v in prev_weights.items()}
                    weights[gap_key] = weights.get(gap_key, 0.0) + alpha
                    
                    sensitivity = post_days / gap_days
                else:
                    # Meeting at the absolute end of the gap
                    r_post = r_pre
                    weights = dict(prev_weights)
                    sensitivity = 0.0

                jump = r_post - r_pre
                
                # Calculate jump weights for hedging: delta between post-weights and pre-weights
                all_keys = set(weights) | set(prev_weights)
                jump_weights = {k: weights.get(k, 0.0) - prev_weights.get(k, 0.0) for k in all_keys}

                records.append({
                    "meeting_date": meeting.isoformat(),
                    "effective_date": eff_date.isoformat(),
                    "gap": f"{start_date} to {end_date}",
                    "r_pre": r_pre,
                    "r_post": r_post,
                    "jump_ois": jump,
                    "sensitivity_ois": sensitivity,
                    "recursive_weights_ois": jump_weights,
                    "is_high_leverage": (0 < post_days < 5)
                })
            else:
                # No meeting in this gap, rate remains stable
                r_post = r_pre
                weights = dict(prev_weights)

            prev_post = r_post
            prev_weights = weights

        return records
