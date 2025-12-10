# core/markets.py
from __future__ import annotations
from typing import List, Tuple, Dict
import pandas as pd

# --- Import do entsoe-py com fallback seguro ---
try:
    from entsoe import EntsoePandasClient, CountryCode  # type: ignore
    _ENTSOE_AVAILABLE = True
except Exception:
    # Pacote não instalado ou bloqueado no ambiente
    EntsoePandasClient = None  # type: ignore
    CountryCode = None         # type: ignore
    _ENTSOE_AVAILABLE = False


def entsoe_available() -> bool:
    """Indica se o entsoe-py está disponível para uso da API."""
    return _ENTSOE_AVAILABLE


# Lista para o seletor
COUNTRY_CHOICES: list[tuple[str, str]] = [
    ("Portugal", "PT"), ("Spain", "ES"), ("France", "FR"),
    ("Germany (DE-LU)", "DE"), ("Netherlands", "NL"),
    ("Belgium", "BE"), ("Italy", "IT"), ("Poland", "PL"),
    ("Austria", "AT"), ("Czechia", "CZ"), ("Hungary", "HU"),
    ("Slovakia", "SK"), ("Slovenia", "SI"), ("Croatia", "HR"),
    ("Greece", "GR"), ("Finland", "FI"), ("Sweden", "SE"),
    ("Norway", "NO"), ("Denmark", "DK"), ("Lithuania", "LT"),
    ("Latvia", "LV"), ("Estonia", "EE"), ("Ireland", "IE"),
]

# Presets (valores de referência; ajuste conforme o DSO/mercado)
COUNTRY_TARIFF_PRESETS: Dict[str, Dict[str, float]] = {
    "PT": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "ES": {"import_fee": 6.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "FR": {"import_fee": 8.0, "export_fee": 1.5, "P_imp": 200.0, "P_exp": 200.0},
    "DE": {"import_fee": 9.0, "export_fee": 2.0, "P_imp": 200.0, "P_exp": 200.0},
    "NL": {"import_fee": 8.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "BE": {"import_fee": 8.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "IT": {"import_fee": 10.0, "export_fee": 2.0, "P_imp": 200.0, "P_exp": 200.0},
    "PL": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "AT": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "CZ": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "HU": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "SK": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "SI": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "HR": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "GR": {"import_fee": 8.0, "export_fee": 1.5, "P_imp": 200.0, "P_exp": 200.0},
    "FI": {"import_fee": 6.0, "export_fee": 0.5, "P_imp": 200.0, "P_exp": 200.0},
    "SE": {"import_fee": 6.0, "export_fee": 0.5, "P_imp": 200.0, "P_exp": 200.0},
    "NO": {"import_fee": 5.0, "export_fee": 0.5, "P_imp": 200.0, "P_exp": 200.0},
    "DK": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "LT": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "LV": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "EE": {"import_fee": 7.0, "export_fee": 1.0, "P_imp": 200.0, "P_exp": 200.0},
    "IE": {"import_fee": 8.0, "export_fee": 1.5, "P_imp": 200.0, "P_exp": 200.0},
}

def get_country_defaults(alpha2: str) -> Dict[str, float]:
    return COUNTRY_TARIFF_PRESETS.get(alpha2, {"import_fee": 0.0, "export_fee": 0.0, "P_imp": 200.0, "P_exp": 200.0})


def _to_utc_15s(df: pd.DataFrame, col_time="datetime", col_price="price_EUR_per_MWh") -> pd.DataFrame:
    out = df.copy()
    out[col_time] = pd.to_datetime(out[col_time], utc=True)
    out = out.set_index(col_time).sort_index()
    out = out.resample("15S").ffill()
    out = out.reset_index()
    out.rename(columns={"index": "datetime"}, inplace=True)
    return out[["datetime", col_price]]


def fetch_entsoe_day_ahead_prices(country_alpha2: str, start_date: str, end_date: str, token: str) -> pd.DataFrame:
    """
    Retorna DataFrame (UTC, 15 s): datetime, price_EUR_per_MWh.
    Lança erro amigável se `entsoe-py` não estiver disponível.
    """
    if not _ENTSOE_AVAILABLE:
        raise RuntimeError(
            "A API ENTSO-E não está disponível neste ambiente. "
            "Adicione 'entsoe-py' ao requirements.txt e redeploy (ou use CSV de preços)."
        )
    if not token:
        raise ValueError("Informe o token ENTSO-E.")

    cc = getattr(CountryCode, country_alpha2)
    client = EntsoePandasClient(api_key=token)

    tz = "Europe/Brussels"
    start = pd.Timestamp(start_date, tz=tz)
    end   = pd.Timestamp(end_date,   tz=tz) + pd.Timedelta(days=1)

    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=90), end)
        s = client.query_day_ahead_prices(cc, cur, nxt)
        if s is not None and len(s) > 0:
            chunks.append(s)
        cur = nxt

    if not chunks:
        raise RuntimeError("Sem dados ENTSO-E para o período informado.")

    series = pd.concat(chunks).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    df = series.to_frame("price_EUR_per_MWh").reset_index().rename(columns={"index": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert("UTC")
    return _to_utc_15s(df, col_time="datetime", col_price="price_EUR_per_MWh")
