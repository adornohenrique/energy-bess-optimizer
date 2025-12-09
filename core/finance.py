"""
Módulo de funções financeiras para o otimizador Energy + BESS.

Aqui ficam centralizados:
- cálculo de CRF (Capital Recovery Factor)
- CAPEX da bateria (a partir de E_cap e P_cap)
- custo anualizado da bateria
- EBITDA
- ROI
"""


from math import pow


def crf(rate_percent: float, n_years: int) -> float:
    """
    Capital Recovery Factor para anualizar CAPEX.

    rate_percent : taxa de desconto ao ano, em %
    n_years      : vida útil em anos

    Retorna um fator tal que:
        CAPEX_anual = CRF * CAPEX_inicial
    """
    r = rate_percent / 100.0
    if n_years <= 0:
        raise ValueError("n_years deve ser > 0")
    if r == 0:
        return 1.0 / n_years
    return r * pow(1 + r, n_years) / (pow(1 + r, n_years) - 1)


def bess_capex_eur(
    E_cap_MWh: float,
    P_cap_MW: float,
    c_E_capex_eur_per_kwh: float,
    c_P_capex_eur_per_kw: float,
) -> float:
    """
    CAPEX da bateria a partir de:
      - E_cap_MWh           : capacidade de energia (MWh)
      - P_cap_MW            : potência (MW)
      - c_E_capex_eur_per_kwh : custo em EUR/kWh
      - c_P_capex_eur_per_kw  : custo em EUR/kW

    Obs: converte kWh -> MWh internamente.
    """
    if E_cap_MWh < 0 or P_cap_MW < 0:
        raise ValueError("E_cap_MWh e P_cap_MW devem ser >= 0")

    # converte EUR/kWh -> EUR/MWh
    c_E_per_MWh = c_E_capex_eur_per_kwh * 1000.0
    # converte EUR/kW -> EUR/MW
    c_P_per_MW = c_P_capex_eur_per_kw * 1000.0

    return c_E_per_MWh * E_cap_MWh + c_P_per_MW * P_cap_MW


def annualized_bess_cost_eur(
    bess_capex_eur_value: float,
    rate_percent: float,
    n_years: int,
) -> float:
    """
    Custo anualizado da bateria (CAPEX transformado em custo ao ano).
    """
    factor = crf(rate_percent, n_years)
    return factor * bess_capex_eur_value


def ebitda_annual_eur(
    revenue_eur: float,
    grid_energy_cost_eur: float,
    bess_annual_cost_eur: float,
    extra_opex_eur: float = 0.0,
) -> float:
    """
    EBITDA simplificado:

      EBITDA = Receita - Custo_energia_rede - Custo_anual_BESS - OPEX_extra
    """
    return revenue_eur - grid_energy_cost_eur - bess_annual_cost_eur - extra_opex_eur


def roi_from_ebitda(
    ebitda_annual_eur: float,
    total_capex_eur: float,
) -> float:
    """
    ROI contábil anual:

      ROI = EBITDA_anual / CAPEX_total

    Retorna em fração (0.15 = 15%).
    """
    if total_capex_eur <= 0:
        return 0.0
    return ebitda_annual_eur / total_capex_eur
