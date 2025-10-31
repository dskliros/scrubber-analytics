import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================================
#      SCRUBBER ANALYTICS DASHBOARD
# ======================================

# ---------- Physical Constants ----------
M_SO2 = 64.066
M_S = 32.06
SO2_FACTOR = M_SO2 / M_S  # ≈ 2.0

# ---------- Helper Functions ----------
def eta_of_deltaT(deltaT, eta_max, k):
    """Filter efficiency as a function of ΔT."""
    return eta_max * (1 - np.exp(-k * deltaT))

def deltaT_required(m_SO2_prod, allowed_conc, m_exh, eta_max, k):
    """Minimum ΔT required to meet emission constraint."""
    m_allowed = allowed_conc * m_exh
    if m_SO2_prod <= m_allowed:
        return 0.0
    eta_req = 1 - (m_allowed / m_SO2_prod)
    eta_req = min(max(eta_req, 0.0), eta_max)
    val = 1 - (eta_req / eta_max)
    val = max(val, 1e-12)
    return float(-np.log(val) / k)

def filter_cost_per_kg(deltaT, m_exh, cp_kJ_per_kgK, elec_price_per_kWh):
    """Filter cooling cost (€/kg fuel)."""
    Q_kJ = m_exh * cp_kJ_per_kgK * deltaT
    Q_kWh = Q_kJ / 3600.0
    return Q_kWh * elec_price_per_kWh

def profit_per_kg(fuel_cost, filter_cost, energy_kWh_per_kg, price_energy_per_kWh):
    """Net profit per kg fuel (€/kg)."""
    revenue = energy_kWh_per_kg * price_energy_per_kWh
    return revenue - fuel_cost - filter_cost

# ---------- Streamlit Config ----------
st.set_page_config(
    page_title="Scrubber Analytics",
    layout="wide",
)

st.title("Scrubber Analytics")
st.caption("Interactive SO₂ Emission Control & Profit Optimization Dashboard")

# ---------- Sidebar Controls ----------
st.sidebar.header("System Parameters")

# Emission & economic parameters
allowed_conc = st.sidebar.slider("Max SO₂ fraction in exhaust", 0.001, 0.01, 0.001, step=0.0005, format="%.4f")
m_exh = st.sidebar.number_input("Exhaust mass (kg/kg fuel)", 1.0, 10.0, 3.0)
cp_kJ_per_kgK = st.sidebar.number_input("Specific heat (kJ/kg·K)", 0.5, 2.0, 1.0)
elec_price_per_kWh = st.sidebar.number_input("Electricity price (€/kWh)", 0.05, 0.5, 0.20)
energy_kWh_per_kg = st.sidebar.number_input("Fuel energy content (kWh/kg)", 5.0, 20.0, 12.0)
price_energy_per_kWh = st.sidebar.number_input("Energy sale price (€/kWh)", 0.05, 0.5, 0.15)

st.sidebar.header("Filter Parameters")
eta_max = st.sidebar.slider("Max filter efficiency ηₘₐₓ", 0.5, 1.0, 0.95)
k = st.sidebar.slider("Filter rate constant k (1/K)", 0.01, 0.2, 0.05)

# ---------- Fuel Data ----------
st.subheader("Fuel Dataset")
default_fuels = pd.DataFrame([
    {"Fuel": "Clean_A", "Sulfur_%": 0.10, "Cost_€/kg": 0.60},
    {"Fuel": "Med_B",   "Sulfur_%": 0.50, "Cost_€/kg": 0.50},
    {"Fuel": "Dirty_C", "Sulfur_%": 1.50, "Cost_€/kg": 0.35},
])

fuels = st.data_editor(
    default_fuels,
    num_rows="dynamic",
    use_container_width=True,
    key="fuel_editor",
)

# ---------- Core Computation ----------
rows = []
for _, row in fuels.iterrows():
    if pd.isna(row["Sulfur_%"]) or pd.isna(row["Cost_€/kg"]):
        continue

    S_pct = row["Sulfur_%"]
    fuel_cost = row["Cost_€/kg"]
    m_SO2_prod = (S_pct / 100.0) * SO2_FACTOR

    ΔT_min = deltaT_required(m_SO2_prod, allowed_conc, m_exh, eta_max, k)
    η_min = eta_of_deltaT(ΔT_min, eta_max, k)
    m_SO2_emit = m_SO2_prod * (1 - η_min)
    filter_cost = filter_cost_per_kg(ΔT_min, m_exh, cp_kJ_per_kgK, elec_price_per_kWh)
    profit = profit_per_kg(fuel_cost, filter_cost, energy_kWh_per_kg, price_energy_per_kWh)

    rows.append({
        "Fuel": row["Fuel"],
        "Sulfur_%": S_pct,
        "SO₂_produced (kg/kg fuel)": m_SO2_prod,
        "ΔT_required (K)": ΔT_min,
        "η_at_min": η_min,
        "SO₂_emitted (kg/kg fuel)": m_SO2_emit,
        "Filter_cost (€/kg fuel)": filter_cost,
        "Profit (€/kg fuel)": profit,
    })

results = pd.DataFrame(rows)

# ---------- Results Table ----------
if not results.empty:
    results = results.sort_values(by="Profit (€/kg fuel)", ascending=False)
    st.subheader("Results Summary")
    st.dataframe(results.style.format(precision=4), use_container_width=True)

    best = results.iloc[0]
    st.success(
        f"**Optimal configuration:** {best['Fuel']} — "
        f"ΔT_required = {best['ΔT_required (K)']:.1f} K, "
        f"Profit = {best['Profit (€/kg fuel)']:.4f} €/kg"
    )

    # ---------- Visualization ----------
    deltaT_grid = np.linspace(0, 200, 401)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Profit vs. Cooling ΔT", "SO₂ Emissions vs. Cooling ΔT"),
        horizontal_spacing=0.12
    )

    for _, r in fuels.iterrows():
        m_SO2_prod = (r["Sulfur_%"] / 100.0) * SO2_FACTOR
        profits, emissions = [], []
        
        for ΔT in deltaT_grid:
            η = eta_of_deltaT(ΔT, eta_max, k)
            m_emit = m_SO2_prod * (1 - η)
            m_allowed = allowed_conc * m_exh
            
            if m_emit > m_allowed:
                profits.append(None)
            else:
                f_cost = filter_cost_per_kg(ΔT, m_exh, cp_kJ_per_kgK, elec_price_per_kWh)
                profits.append(profit_per_kg(
                    r["Cost_€/kg"], f_cost,
                    energy_kWh_per_kg, price_energy_per_kWh
                ))
            emissions.append(m_emit)
        
        # Add profit trace
        fig.add_trace(
            go.Scatter(
                x=deltaT_grid,
                y=profits,
                mode='lines',
                name=r["Fuel"],
                legendgroup=r["Fuel"],
                hovertemplate='ΔT: %{x:.1f} K<br>Profit: %{y:.4f} €/kg<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add emissions trace
        fig.add_trace(
            go.Scatter(
                x=deltaT_grid,
                y=emissions,
                mode='lines',
                name=r["Fuel"],
                legendgroup=r["Fuel"],
                showlegend=False,
                hovertemplate='ΔT: %{x:.1f} K<br>SO₂: %{y:.6f} kg/kg<extra></extra>'
            ),
            row=1, col=2
        )

    # Add emission limit line
    fig.add_trace(
        go.Scatter(
            x=[0, 200],
            y=[allowed_conc * m_exh, allowed_conc * m_exh],
            mode='lines',
            name='Emission Limit',
            line=dict(color='red', dash='dash'),
            hovertemplate='Limit: %{y:.6f} kg/kg<extra></extra>'
        ),
        row=1, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Cooling ΔT (K)", row=1, col=1)
    fig.update_xaxes(title_text="Cooling ΔT (K)", row=1, col=2)
    fig.update_yaxes(title_text="Profit (€/kg fuel)", row=1, col=1)
    fig.update_yaxes(title_text="SO₂ emitted (kg/kg fuel)", row=1, col=2)

    # Update layout
    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please enter at least one valid fuel entry.")

# ---------- Footer ----------
st.markdown("---")
st.caption("Developed for process optimization and emission control — © 2025 Scrubber Analytics")
