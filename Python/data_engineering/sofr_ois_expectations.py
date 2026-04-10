from __future__ import annotations

import bisect
from datetime import date, timedelta
import calendar
import math
import re
import warnings
from typing import Any, Iterable
from dataclasses import dataclass
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd


@dataclass
class OISTenor:
    unit: str   # 'D', 'W', 'M', 'Y'
    value: int


def year_fraction_act360(d0: date, d1: date) -> float:
    return max((d1 - d0).days, 0) / 360.0


def month_start(d: date) -> date:
    return d.replace(day=1)


def add_years(d: date, years: int) -> date:
    return add_months(d, years * 12)


def add_days(d: date, days: int) -> date:
    return d + timedelta(days=days)


def months_between(d0: date, d1: date) -> int:
    months = (d1.year - d0.year) * 12 + (d1.month - d0.month)
    if add_months(d0, months) > d1:
        months -= 1
    return max(months, 0)


def parse_ois_tenor(code: str) -> OISTenor | None:
    if not isinstance(code, str) or not code.startswith("USDSROIS"):
        return None

    if code == "USDSROISON=":
        return OISTenor("D", 1)

    if code == "USDSROISSW=":
        return OISTenor("W", 1)

    match = re.match(r"USDSROIS(\d+)([WMY])=", code)
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2)

    return OISTenor(unit, value)


def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))


def adjust_modified_following(d: date) -> date:
    if _is_us_business_day(d):
        return d
    # move forward to Monday
    forward = d
    while not _is_us_business_day(forward):
        forward += timedelta(days=1)
    # if month changed, move backward instead
    if forward.month != d.month:
        backward = d
        while not _is_us_business_day(backward):
            backward -= timedelta(days=1)
        return backward
    return forward


def tenor_to_maturity(valuation_date: date, tenor: OISTenor) -> date:
    # spot lag T+2 business days
    spot = _add_us_business_days(valuation_date, 2)

    if tenor.unit == "D":
        maturity = spot + timedelta(days=tenor.value)
    elif tenor.unit == "W":
        maturity = spot + timedelta(days=7 * tenor.value)
    elif tenor.unit == "M":
        maturity = add_months(spot, tenor.value)
    elif tenor.unit == "Y":
        maturity = add_months(spot, 12 * tenor.value)
    else:
        return None

    return adjust_modified_following(maturity)

US_HOLIDAY_CALENDAR = USFederalHolidayCalendar()


def _is_us_business_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    ts = pd.Timestamp(d)
    try:
        holidays = US_HOLIDAY_CALENDAR.holidays(start=ts, end=ts)
    except TypeError:
        holidays = US_HOLIDAY_CALENDAR.holidays(ts, ts)
    return len(holidays) == 0


def _add_us_business_days(d: date, business_days: int) -> date:
    out = d
    added = 0
    while added < business_days:
        out += timedelta(days=1)
        if _is_us_business_day(out):
            added += 1
    return out


def _business_days_between(start: date, end_exclusive: date) -> int:
    if end_exclusive <= start:
        return 0
    holidays = US_HOLIDAY_CALENDAR.holidays(
        start=start,
        end=end_exclusive - timedelta(days=1),
    )
    return int(
        np.busday_count(
            np.datetime64(start),
            np.datetime64(end_exclusive),
            holidays=holidays.to_numpy(dtype="datetime64[D]"),
        )
    )


def _third_wednesday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 2:
        d += timedelta(days=1)
    return d + timedelta(days=14)


def _imm_on_or_after(d: date) -> date:
    imm_months = (3, 6, 9, 12)
    y = d.year
    while True:
        for m in imm_months:
            imm = _third_wednesday(y, m)
            if imm >= d:
                return imm
        y += 1


def _next_imm_date(d: date) -> date:
    imm_months = (3, 6, 9, 12)
    y = d.year
    while True:
        for m in imm_months:
            imm = _third_wednesday(y, m)
            if imm > d:
                return imm
        y += 1


class SR1MatrixEstimator:

    def __init__(
        self,
        valuation_date: date,
        monthly_rates: dict[str, float],  # keys "YYYY-MM", "SR1:YYYY-MM", or "SR3:YYYY-MM-DD"
        meetings: Iterable[date],
    ) -> None:
        self.valuation_date = valuation_date
        self.instrument_rates = {
            k: float(v) for k, v in sorted(monthly_rates.items())
        }
        self.meetings = sorted([m for m in meetings if m >= valuation_date])

    def estimate(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:

        if not self.instrument_rates or not self.meetings:
            return [], {}

        instruments: list[tuple[str, date, date, float]] = []
        for instrument_name, rate in self.instrument_rates.items():
            if instrument_name.startswith("SR3:"):
                start = _imm_on_or_after(date.fromisoformat(instrument_name.split(":", 1)[1]))
                end = _next_imm_date(start)
            else:
                month_str = instrument_name.split(":", 1)[1] if instrument_name.startswith("SR1:") else instrument_name
                year, month_num = map(int, month_str.split("-"))
                start = date(year, month_num, 1)
                end = add_months(start, 1)
            instruments.append((instrument_name, start, end, rate))

        n_instruments = len(instruments)
        n_meetings = len(self.meetings)

        A = np.zeros((n_instruments, n_meetings + 1))  # +1 for r0
        b = np.zeros(n_instruments)
        instrument_names = [name for name, _, _, _ in instruments]

        for i, (_, start, end, rate) in enumerate(instruments):

            b[i] = rate
            total_days = _business_days_between(start, end)
            if total_days <= 0:
                continue

            # r0 exposure
            A[i, 0] = 1.0

            for j, meeting in enumerate(self.meetings):
                eff = meeting + timedelta(days=1)
                exposed_days = _business_days_between(max(eff, start), end)

                if exposed_days > 0:
                    A[i, j + 1] = exposed_days / total_days

        # Ridge regularization
        lambda_reg = 1e-6
        I = np.eye(n_meetings + 1)
        I[0, 0] = 0.0  # don't regularize r0

        # Regularized system
        A_reg = np.vstack([A, lambda_reg * I])
        b_reg = np.concatenate([b, np.zeros(n_meetings + 1)])

        solution, *_ = np.linalg.lstsq(A_reg, b_reg, rcond=None)

        # Projection matrix mapping instruments -> parameters
        P = np.linalg.pinv(A_reg)

        r0 = solution[0]
        jumps = solution[1:]

        # Only keep columns corresponding to real instruments
        P_instr = P[:, :n_instruments]

        # Portfolio weights for each meeting jump
        portfolio_weights = {
            meeting.isoformat(): {
                instrument_names[i]: float(P_instr[j + 1, i])
                for i in range(n_instruments)
            }
            for j, meeting in enumerate(self.meetings)
        }

        # Build meeting path
        records = []
        cumulative = r0

        for j, meeting in enumerate(self.meetings):
            jump = jumps[j]

            r_pre = cumulative
            r_post = cumulative + jump
            cumulative = r_post

            records.append({
                "meeting_date": meeting.isoformat(),
                "effective_date": (meeting + timedelta(days=1)).isoformat(),
                "r_pre": r_pre,
                "r_post": r_post,
                "jump": jump,
                "jump_sr1": jump,
            })

        return records, portfolio_weights


class OISMatrixEstimator:
    """
    Stable matrix-based estimator of FOMC jumps from OIS curve.
    """

    def __init__(
        self,
        valuation_date: date,
        ois_quotes: list[dict[str, float]],
        meetings: Iterable[date],
    ) -> None:
        self.valuation_date = valuation_date
        self.ois_quotes = ois_quotes
        self.meetings = sorted(m for m in meetings if m >= valuation_date)

    def _tenor_to_date(self, tenor_months: float) -> date:
        if tenor_months < 1.0:
            days = max(int(round(tenor_months * 30.0)), 1)
            return self.valuation_date + timedelta(days=days)

        d = self.valuation_date
        y = d.year + (d.month - 1 + int(round(tenor_months))) // 12
        m = (d.month - 1 + int(round(tenor_months))) % 12 + 1
        last_day = calendar.monthrange(y, m)[1]
        return date(y, m, min(d.day, last_day))

    def estimate(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
        if not self.ois_quotes or not self.meetings:
            return [], {}

        instruments: list[tuple[str, date, int, float]] = []
        max_maturity = add_months(self.valuation_date, 6)
        for idx, row in enumerate(self.ois_quotes):
            tenor = float(row["tenor_months"])
            if tenor <= 0:
                continue

            maturity = self._tenor_to_date(tenor)
            if maturity <= self.valuation_date:
                continue
            if maturity > max_maturity:
                continue
            has_relevant_meeting = any(
                self.valuation_date < (meeting + timedelta(days=1)) < maturity
                for meeting in self.meetings
            )
            if not has_relevant_meeting:
                continue

            total_days = _business_days_between(self.valuation_date, maturity)
            if total_days <= 0:
                continue

            instruments.append((f"OIS_{tenor:g}M_{idx}", maturity, total_days, float(row["rate"])))

        if len(instruments) == 0:
            return [], {}

        n_points = len(instruments)
        n_meetings = len(self.meetings)

        A = np.zeros((n_points, n_meetings + 1))  # +1 for r0
        b = np.array([inst[3] for inst in instruments])
        instrument_names = [inst[0] for inst in instruments]

        for i, (_, maturity, total_days, _) in enumerate(instruments):

            # r0 exposure
            A[i, 0] = 1.0

            for j, meeting in enumerate(self.meetings):
                eff = meeting + timedelta(days=1)
                exposed_days = _business_days_between(max(eff, self.valuation_date), maturity)
                if exposed_days > 0:
                    A[i, j + 1] = exposed_days / total_days

        lambda_reg = 1e-6
        I = np.eye(n_meetings + 1)
        I[0, 0] = 0.0  # don't regularize r0

        A_reg = np.vstack([A, lambda_reg * I])
        b_reg = np.concatenate([b, np.zeros(n_meetings + 1)])

        solution, *_ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
        P = np.linalg.pinv(A_reg)
        P_instr = P[:, :n_points]

        r0 = solution[0]
        jumps = solution[1:]

        # Build output
        records = []
        cumulative = r0
        portfolio_weights = {
            meeting.isoformat(): {
                instrument_names[i]: float(P_instr[j + 1, i])
                for i in range(len(instrument_names))
            }
            for j, meeting in enumerate(self.meetings)
        }

        for j, meeting in enumerate(self.meetings):
            jump = jumps[j]
            r_pre = cumulative
            r_post = cumulative + jump
            cumulative = r_post

            records.append({
                "meeting_date": meeting.isoformat(),
                "effective_date": (meeting + timedelta(days=1)).isoformat(),
                "r_pre": r_pre,
                "r_post": r_post,
                "jump": jump,
                "jump_ois": jump,
            })

        return records, portfolio_weights
