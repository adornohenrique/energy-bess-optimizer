# core/finance.py
import math

def crf(rate_percent: float, n_years: int) -> float:
    r = float(rate_percent) / 100.0
    n = int(n_years)
    if n <= 0:
        return 1.0
    if abs(r) < 1e-12:
        return 1.0 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def annualized_cost(capex_eur: float, rate_percent: float, n_years: int) -> float:
    return float(capex_eur) * crf(rate_percent, n_years)

def lcoe_annual(capex_total_eur: float,
                rate_percent: float,
                n_years: int,
                energy_annual_MWh: float,
                opex_fix_eur_per_year: float = 0.0,
                opex_var_eur_per_MWh: float = 0.0) -> float:
    e = max(1e-9, float(energy_annual_MWh))
    capex_ann = annualized_cost(capex_total_eur, rate_percent, n_years)
    return (capex_ann + float(opex_fix_eur_per_year)) / e + float(opex_var_eur_per_MWh)

def bess_capex_eur(E_cap_MWh: float, P_cap_MW: float,
                   c_E_eur_per_kWh: float, c_P_eur_per_kW: float) -> float:
    # capex = E (kWh)*€/kWh + P (kW)*€/kW
    return float(E_cap_MWh) * 1000.0 * float(c_E_eur_per_kWh) + float(P_cap_MW) * 1000.0 * float(c_P_eur_per_kW)
