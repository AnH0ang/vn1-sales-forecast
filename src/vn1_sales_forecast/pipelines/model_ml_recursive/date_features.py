from datetime import date, timedelta

import numpy as np
import polars as pl
from holidays import country_holidays


def number_of_holidays(dates: pl.Expr) -> pl.Expr:
    def _get_nb_hollidays(d: date) -> int:
        countries = country_holidays("US", years=list(range(2020, 2024)))
        ds = [d + timedelta(o) for o in range(7)]
        hs = {h for d in ds if (h := countries.get(d)) is not None}
        return len(hs)

    return dates.map_elements(_get_nb_hollidays, return_dtype=pl.Int64)


def is_chrismas(dates: pl.Expr) -> pl.Expr:
    def _get_nb_hollidays(d: date) -> int:
        ds = [d + timedelta(o) for o in range(7)]
        hs = {d for d in ds if (d.day == 24 and d.month == 12)}
        return len(hs)

    return dates.map_elements(_get_nb_hollidays, return_dtype=pl.Int64)


def is_back_friday(dates: pl.Expr) -> pl.Expr:
    countries = country_holidays("US", years=list(range(2020, 2024)))

    def _get_nb_hollidays(d: date) -> int:
        ds = [d + timedelta(o + 1) for o in range(7)]
        hs = {countries.get(d) == "Thanksgiving" for d in ds}
        return len(hs)

    return dates.map_elements(_get_nb_hollidays, return_dtype=pl.Int64)


def fourier_term(sin: bool = True, K: int = 1):
    def _inner(date: pl.Expr) -> pl.Expr:
        t = date.dt.week() / 52
        v = 2 * np.pi * K * t
        r = v.sin() if sin else v.cos()
        return r

    _inner.__name__ = f"fourier_term_{K}{'sin' if sin else 'cos'}"
    return _inner
