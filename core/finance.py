"""
Funções financeiras e utilitários para o otimizador Energy + BESS.
Sprint 2 mantém as mesmas assinaturas do Sprint 1.
"""

from math import pow


def crf(rate_percent: float, n_years: int) -> float:
    """
    Capital Recovery Factor para anualizar CAPEX.
    Retorna fator tal que CAPEX_anual = CRF * CAPEX_inicial.
    """
    if n_years <= 0:
        raise ValueError("n_years deve ser > 0")
    r = rate_percent / 100.0
    if r == 0:
        return 1.0 / n_years
    return r * pow(1 + r, n_years) / (pow(1 + r, n_years) - 1)


def bess_capex_eur(E_cap_MWh: float, P_cap_MW: float,
                   c_E_capex_eur_per_kwh: float,
                   c_P_capex_eur_per_kw: float) -> float:
    """
    CAPEX do BESS = 1000*c_E*E_cap(MWh) + 1000*c_P*P_cap(MW).
    """
    if E_cap_MWh < 0 or P_cap_MW < 0:
        raise ValueError("E_cap_MWh e P_cap_MW devem ser >= 0")
    c_E_per_MWh = c_E_capex_eur_per_kwh * 1000.0
    c_P_per_MW = c_P_capex_eur_per_kw * 1000.0
    return c_E_per_MWh * E_cap_MWh + c_P_per_MW * P_cap_MW


def annualized_cost(capex_eur: float, rate_percent: float, n_years: int) -> float:
    """
    CAPEX anualizado.
    """
    return crf(rate_percent, n_years) * capex_eur


def lcoe_annual(capex_total_eur: float,
                rate_percent: float,
                n_years: int,
                energy_annual_MWh: float,
                opex_fix_eur_per_year: float = 0.0,
                opex_var_eur_per_MWh: float = 0.0) -> float:
    """
    LCOE = (CRF*CAPEX_total + OPEX_fix + OPEX_var*E_anual) / E_anual
    """
    if energy_annual_MWh <= 0:
        return float("inf")
    annual_capex = annualized_cost(capex_total_eur, rate_percent, n_years)
    annual_cost = annual_capex + opex_fix_eur_per_year + opex_var_eur_per_MWh * energy_annual_MWh
    return annual_cost / energy_annual_MWh
