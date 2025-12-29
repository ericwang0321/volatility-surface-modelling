import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from src.vol_surface import VolatilitySurface
from src.pricer import MonteCarloPricer

st.set_page_config(page_title="Quant Volatility Surface", layout="wide")

# Title and Description
st.title("📈 Industrial Volatility Surface & Pricing Engine")

# --- 1. Sidebar Configuration & Navigation ---
st.sidebar.header("Global Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
build_btn = st.sidebar.button("Build Surface 🚀")

st.sidebar.markdown("---")
# FIX: Use Radio button for navigation to prevent "Tab Jumping" on re-runs
nav_option = st.sidebar.radio(
    "Select Module", 
    ["1. 3D Volatility Surfaces", "2. Exotic Pricing Engine", "3. Smile Calibration", "4. Raw Data Inspector"]
)

# --- 2. State Management ---
# Build Surface and persist in session_state
if build_btn:
    with st.spinner(f"Fetching data and calibrating models for {ticker}..."):
        try:
            surface = VolatilitySurface(ticker)
            surface.build()
            st.session_state['vol_surface_obj'] = surface
            st.success(f"Successfully calibrated {len(surface.svi_params)} expiration slices!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- 3. Main Logic ---
if 'vol_surface_obj' in st.session_state:
    surface = st.session_state['vol_surface_obj']
    
    # ---------------------------------------------------------
    # Module 1: 3D Surfaces
    # ---------------------------------------------------------
    if nav_option == "1. 3D Volatility Surfaces":
        st.subheader("Implied vs. Local Volatility Topology")
        
        # Get Grid
        X, Y, Z_imp, Z_loc = surface.get_mesh_grid()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption("Market Implied Volatility (Expectation)")
            fig_imp = go.Figure(data=[go.Surface(z=Z_imp, x=X, y=Y, colorscale='Viridis')])
            fig_imp.update_layout(scene=dict(xaxis_title='Moneyness', yaxis_title='Time', zaxis_title='Vol'), 
                                margin=dict(l=10, r=10, b=10, t=10), height=500)
            st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            st.caption("Dupire Local Volatility (Pricing Input)")
            fig_loc = go.Figure(data=[go.Surface(z=Z_loc, x=X, y=Y, colorscale='Turbo')])
            fig_loc.update_layout(scene=dict(xaxis_title='Spot/Strike', yaxis_title='Time', zaxis_title='Loc Vol'), 
                                margin=dict(l=10, r=10, b=10, t=10), height=500)
            st.plotly_chart(fig_loc, use_container_width=True)

    # ---------------------------------------------------------
    # Module 2: Exotic Pricing Engine (FIXED)
    # ---------------------------------------------------------
    elif nav_option == "2. Exotic Pricing Engine":
        st.subheader("📉 Barrier Option Pricer (Down-and-Out Call)")
        
        # Inputs
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            S0 = surface.spot_price
            st.metric("Spot Price (S0)", f"${S0:.2f}")
            strike = st.number_input("Strike Price (K)", value=float(int(S0 * 1.05)))
            
        with col_p2:
            T_years = st.number_input("Time to Maturity (Years)", value=1.0)
            barrier = st.number_input("Barrier Level (Knock-out)", value=float(int(S0 * 0.85)))
            
        with col_p3:
            r_rate = st.number_input("Risk Free Rate", value=0.045)
            n_sims = st.slider("Monte Carlo Paths", 1000, 5000, 2000)

        # Run Button
        if st.button("Run Monte Carlo Simulation 🎲"):
            pricer = MonteCarloPricer(S0, r_rate, T_years, surface)
            
            with st.spinner("Simulating paths... (This uses Local Vol surface lookup)"):
                # 1. Benchmark BS
                atm_vol = surface.get_implied_vol(0, T_years)
                res_bs = pricer.price_barrier_option(strike, barrier, model="black_scholes", const_vol=atm_vol, n_paths=n_sims)
                
                # 2. Local Vol
                res_lv = pricer.price_barrier_option(strike, barrier, model="local_vol", n_paths=n_sims)
                
                # Store results in session state to prevent disappearance
                st.session_state['pricing_res'] = (res_bs, res_lv, atm_vol)

        # Display Results if they exist
        if 'pricing_res' in st.session_state:
            res_bs, res_lv, atm_vol = st.session_state['pricing_res']
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("ATM Volatility (BS Input)", f"{atm_vol:.2%}")
            c2.metric("Black-Scholes Price", f"${res_bs['price']:.2f}")
            
            diff = res_lv['price'] - res_bs['price']
            color = "normal" if diff < 0 else "inverse" # Green if cheaper, Red if expensive
            c3.metric("Local Vol Price", f"${res_lv['price']:.2f}", delta=f"{diff:.2f}", delta_color=color)

            # Dynamic Analysis Text
            st.info("💡 **Model Risk Analysis:**")
            
            if res_lv['price'] < res_bs['price']:
                st.markdown(f"""
                **Result: Local Vol < Black-Scholes.** This is the classic "Skew Effect" for Down-and-Out Calls.
                * The Local Volatility surface implies higher volatility on the downside (near the barrier).
                * This **increases the probability of hitting the barrier** compared to the constant ATM volatility assumption.
                * **Conclusion:** The Black-Scholes model is likely **overpricing** this option by ignoring the skew risk.
                """)
            else:
                st.markdown(f"""
                **Result: Local Vol > Black-Scholes.** This indicates a "Vega Dominance" scenario.
                * While the barrier risk exists, the Local Volatility model might be picking up **significantly higher volatility** along the surviving paths (Upside or Recovery).
                * Notice the spikes in your Local Vol surface? If the path passes through those high-vol zones, the option accumulates more value (Vega) than the barrier risk takes away.
                * **Conclusion:** The Black-Scholes model might be **underpricing** the potential volatility of the asset if it stays alive.
                """)

    # ---------------------------------------------------------
    # Module 3: Smile Calibration
    # ---------------------------------------------------------
    elif nav_option == "3. Smile Calibration":
        st.subheader("SVI Fit Inspection")
        available_expiries = sorted(surface.svi_params.keys())
        if available_expiries:
            selected_T = st.selectbox("Select Expiration (Years)", available_expiries)
            if selected_T:
                slice_data = surface.raw_data[surface.raw_data['T'] == selected_T]
                k_market = np.log(slice_data['moneyness'])
                vol_market = slice_data['impliedVolatility']
                
                k_grid = np.linspace(k_market.min()-0.1, k_market.max()+0.1, 100)
                vol_model = [surface.get_implied_vol(k, selected_T) for k in k_grid]
                m_grid = np.exp(k_grid)
                
                fig_2d = go.Figure()
                fig_2d.add_trace(go.Scatter(x=slice_data['moneyness'], y=vol_market, mode='markers', name='Market Data', marker=dict(color='red', size=8)))
                fig_2d.add_trace(go.Scatter(x=m_grid, y=vol_model, mode='lines', name='SVI Model', line=dict(color='blue')))
                fig_2d.update_layout(title=f"Smile at T={selected_T:.4f}", xaxis_title="Moneyness", yaxis_title="Implied Vol")
                st.plotly_chart(fig_2d, use_container_width=True)

    # ---------------------------------------------------------
    # Module 4: Raw Data
    # ---------------------------------------------------------
    elif nav_option == "4. Raw Data Inspector":
        st.subheader("Cleaned Market Data")
        st.dataframe(surface.raw_data)

else:
    st.info("👈 Click **'Build Surface'** in the sidebar to start.")