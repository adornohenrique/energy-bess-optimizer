# core/markets.py
from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np

# entsoe-py
from entsoe import EntsoePandasClient, CountryCode


# Países/zonas prontos para uso (pode ampliar esta lista quando quiser)
COUNTRY_CHOICES: list[tuple[str, str]] = [
    ("Portugal", "PT"),
    ("Spain", "ES"),
    ("France", "FR"),
    ("Germany (DE-LU)", "DE"),
    ("Netherlands", "NL"),
    ("Belgium", "BE"),
    ("Italy", "IT"),
    ("Poland", "PL"),
    ("Austria", "AT"),
    ("Czechia", "CZ"),
    ("Hungary", "HU"),
    ("Slovakia", "SK"),
    ("Slovenia", "SI"),
    ("Croatia", "HR"),
    ("Greece", "GR"),
    ("Finland", "FI"),
    ("Sweden", "SE"),
    ("Norway", "NO"),
    ("Denmark", "DK"),
    ("Lithuania", "LT"),
    ("Latvia", "LV"),
    ("Estonia", "EE"),
    ("Ireland", "IE"),
]

def _to_utc_15s(df: pd.DataFrame, col_time="datetime", col_price="price_EUR_per_MWh") -> pd.DataFrame:
    out = df.copy()
    out[col_time] = pd.to_datetime(out[col_time], utc=True)
    out = out.set_index(col_time).sort_index()
    # entsoe vem em 60 min — reamostra para 15 s com forward-fill
    out = out.resample("15S").ffill()
    out = out.reset_index()
    out.rename(columns={"index": "datetime"}, inplace=True)
    return out[[ "datetime", col_price ]]

def fetch_entsoe_day_ahead_prices(country_alpha2: str,
                                  start_date: str,
                                  end_date: str,
                                  token: str) -> pd.DataFrame:
    """
    Retorna DataFrame: datetime (UTC) + price_EUR_per_MWh, reamostrado em 15 s.
    start_date, end_date no formato 'YYYY-MM-DD'.
    """
    if not token:
        raise ValueError("Informe o token ENTSO-E.")
    cc = getattr(CountryCode, country_alpha2)
    client = EntsoePandasClient(api_key=token)

    # ENTSO-E exige timezone Europe/Brussels (recomendado)
    tz = "Europe/Brussels"
    start = pd.Timestamp(start_date, tz=tz)
    end   = pd.Timestamp(end_date,   tz=tz) + pd.Timedelta(days=1)  # incluir o fim

    # divide por janelas de até 90 dias para evitar timeouts
    prices = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=90), end)
        s = client.query_day_ahead_prices(cc, cur, nxt)  # Series com tz
        if s is not None and len(s) > 0:
            prices.append(s)
        cur = nxt

    if not prices:
        raise RuntimeError("Sem dados retornados pela ENTSO-E para o período escolhido.")

    series = pd.concat(prices).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    df = series.to_frame("price_EUR_per_MWh").reset_index().rename(columns={"index":"datetime"})
    # converte para UTC e reamostra 15s
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert("UTC")
    df = _to_utc_15s(df, col_time="datetime", col_price="price_EUR_per_MWh")
    return df
