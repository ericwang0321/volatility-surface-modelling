import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from src.vol_surface import VolatilitySurface
from src.pricer import MonteCarloPricer

st.set_page_config(page_title="Quant Volatility Surface", layout="wide")

# Title and Description
st.title("📈 Industrial Volatility Surface & Pricing Engine")

# --- Helper Class for UI Compatibility ---
class SimpleRateProvider:
    """Helper to wrap the UI's manual interest rate input into a provider object."""
    def __init__(self, r):
        self.r = r
    def get_risk_free_rate(self, T):
        return self.r

# --- 1. Sidebar Configuration & Navigation ---
st.sidebar.header("Global Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
build_btn = st.sidebar.button("Build Surface 🚀")

st.sidebar.markdown("---")

# Navigation Menu
nav_option = st.sidebar.radio(
    "Select Module", 
    [
        "1. 3D Volatility Surfaces", 
        "2. Exotic Pricing Engine", 
        "3. Hedging & Greeks Analysis", 
        "4. Smile Calibration", 
        "5. Raw Data Inspector"
    ]
)

# --- 2. State Management ---
if build_btn:
    with st.spinner(f"Fetching data and calibrating models for {ticker}..."):
        try:
            # Re-initialize
            if 'vol_surface_obj' in st.session_state:
                del st.session_state['vol_surface_obj']
            
            surface = VolatilitySurface(ticker)
            surface.build()
            st.session_state['vol_surface_obj'] = surface
            st.success(f"Successfully calibrated {len(surface.svi_params)} expiration slices!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.text(traceback.format_exc())

# --- 3. Main Logic ---
if 'vol_surface_obj' in st.session_state:
    surface = st.session_state['vol_surface_obj']
    
    # ---------------------------------------------------------
    # Module 1: 3D Volatility Surfaces
    # ---------------------------------------------------------
    if nav_option == "1. 3D Volatility Surfaces":
        st.subheader("Implied vs. Local Volatility Topology")
        
        # Get Grid
        # vol_surface.py updated to return X (Spot Moneyness), Y (Time), Z_imp, Z_loc
        try:
            X, Y, Z_imp, Z_loc = surface.get_mesh_grid()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("Market Implied Volatility (Expectation)")
                fig_imp = go.Figure(data=[go.Surface(z=Z_imp, x=X, y=Y, colorscale='Viridis')])
                fig_imp.update_layout(
                    scene=dict(xaxis_title='Moneyness (Spot/Strike)', yaxis_title='Time', zaxis_title='Vol'), 
                    margin=dict(l=10, r=10, b=10, t=10), height=500
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            with col2:
                st.caption("Dupire Local Volatility (Pricing Input)")
                fig_loc = go.Figure(data=[go.Surface(z=Z_loc, x=X, y=Y, colorscale='Turbo')])
                fig_loc.update_layout(
                    scene=dict(xaxis_title='Moneyness (Spot/Strike)', yaxis_title='Time', zaxis_title='Loc Vol'), 
                    margin=dict(l=10, r=10, b=10, t=10), height=500
                )
                st.plotly_chart(fig_loc, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating surface: {e}")

    # ---------------------------------------------------------
    # Module 2: Exotic Pricing Engine
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
            # User Manual Override for Rate
            r_rate = st.number_input("Risk Free Rate (Manual Override)", value=0.045, format="%.4f")
            n_sims = st.slider("Monte Carlo Paths", 1000, 50000, 10000)

        # Run Button
        if st.button("Run Monte Carlo Simulation 🎲"):
            # Prepare dependencies for updated Pricer signature
            # Signature: (S0, T, rate_provider, q, vol_surface)
            manual_rate_provider = SimpleRateProvider(r_rate)
            q_yield = surface.dividend_yield
            
            pricer = MonteCarloPricer(S0, T_years, manual_rate_provider, q_yield, surface)
            
            with st.spinner("Simulating paths... (This uses Local Vol surface lookup)"):
                # 1. Benchmark BS
                # Note: get_implied_vol inputs are (log_spot_moneyness, T)
                # ATM => Strike=Spot => log(1) = 0
                atm_vol = surface.get_implied_vol(0, T_years)
                
                res_bs = pricer.price_barrier_option(
                    strike, barrier, model="black_scholes", const_vol=atm_vol, n_paths=n_sims
                )
                
                # 2. Local Vol
                res_lv = pricer.price_barrier_option(
                    strike, barrier, model="local_vol", n_paths=n_sims
                )
                
                # Store results in session state
                st.session_state['pricing_res'] = (res_bs, res_lv, atm_vol)

        # Display Results
        if 'pricing_res' in st.session_state:
            res_bs, res_lv, atm_vol = st.session_state['pricing_res']
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("ATM Volatility (BS Input)", f"{atm_vol:.2%}")
            c2.metric("Black-Scholes Price", f"${res_bs['price']:.2f}")
            
            diff = res_lv['price'] - res_bs['price']
            # If LV < BS (negative diff), that's expected for Down-and-Out + Negative Skew
            # We color it green (normal) if it drops, or inverse logic depending on preference.
            c3.metric("Local Vol Price", f"${res_lv['price']:.2f}", delta=f"{diff:.2f}")

            st.info("💡 **Model Risk Analysis:**")
            if res_lv['price'] < res_bs['price']:
                st.markdown(f"""
                **Result: Local Vol < Black-Scholes.** This aligns with the **Negative Skew** of Equity markets.
                
                * **Why?** The Local Volatility surface captures that volatility *increases* as the price drops towards the barrier.
                * **Impact:** Higher volatility on the downside = Higher probability of hitting the barrier = Lower probability of survival = **Lower Price**.
                """)
            else:
                st.markdown(f"""
                **Result: Local Vol > Black-Scholes.** This might occur if the barrier is very deep OTM or the volatility surface has a different shape (e.g., inverted skew).
                """)

    # ---------------------------------------------------------
    # Module 3: Hedging & Greeks Analysis
    # ---------------------------------------------------------
    elif nav_option == "3. Hedging & Greeks Analysis":
        st.subheader("🛡️ Hedge Effectiveness Analysis (Delta Profile)")
        st.markdown("""
        **Scenario:** You sold a Down-and-Out Call and need to hedge. 
        This chart compares the **Delta (Hedge Ratio)** calculated by Black-Scholes vs. Local Volatility as the spot price moves.
        """)
        
        # Parameters
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            S_current = surface.spot_price
            barrier_level = st.number_input("Barrier Level", value=float(int(S_current * 0.85)), key="h_bar")
            strike_level = st.number_input("Strike Price", value=float(int(S_current * 1.05)), key="h_str")
        with col_h2:
            T_hedge = st.number_input("Time to Mat", value=1.0, key="h_t")
            n_sims_hedge = st.slider("Simulations per Point", 1000, 20000, 5000, key="h_sims")
            
        if st.button("Calculate Delta Profile 📉"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Range: From below Barrier to ITM
            spot_range = np.linspace(barrier_level * 0.95, S_current * 1.1, 15)
            
            bs_deltas = []
            lv_deltas = []
            
            # Use surface dividend yield
            q_yield = surface.dividend_yield
            # Use internal rate provider from surface for consistency in hedging module
            # (Or could use manual input if desired)
            rate_provider = surface.rate_provider
            
            try:
                for i, s_val in enumerate(spot_range):
                    status_text.text(f"Calculating Delta for Spot Price ${s_val:.2f}...")
                    
                    # Create temporary pricer at hypothetical spot s_val
                    # Updated Signature: (S0, T, rate_provider, q, vol_surface)
                    temp_pricer = MonteCarloPricer(s_val, T_hedge, rate_provider, q_yield, surface)
                    
                    # 1. BS Delta
                    d_bs = temp_pricer.calculate_delta(
                        strike_level, barrier_level, 
                        model="black_scholes", 
                        n_paths=n_sims_hedge
                    )
                    bs_deltas.append(d_bs)
                    
                    # 2. Local Vol Delta
                    d_lv = temp_pricer.calculate_delta(
                        strike_level, barrier_level, 
                        model="local_vol", 
                        n_paths=n_sims_hedge
                    )
                    lv_deltas.append(d_lv)
                    
                    progress_bar.progress((i + 1) / len(spot_range))
                
                status_text.text("Calculation Complete!")
                
                # Plotting
                fig_delta = go.Figure()
                
                fig_delta.add_trace(go.Scatter(
                    x=spot_range, y=bs_deltas, 
                    mode='lines+markers', name='BS Delta', 
                    line=dict(color='gray', dash='dash')
                ))
                
                fig_delta.add_trace(go.Scatter(
                    x=spot_range, y=lv_deltas, 
                    mode='lines+markers', name='Local Vol Delta', 
                    line=dict(color='red', width=3)
                ))
                
                fig_delta.add_vline(x=barrier_level, line_width=2, line_dash="dot", line_color="black", annotation_text="Barrier")
                
                fig_delta.update_layout(
                    title=f"Delta Skew: Hedging Ratio vs. Spot Price (N_SIMS={n_sims_hedge})",
                    xaxis_title="Spot Price",
                    yaxis_title="Option Delta",
                    hovermode="x unified",
                    height=600
                )
                st.plotly_chart(fig_delta, use_container_width=True)
                
                st.info("""
                **Quant Insight:**
                The **Local Vol Delta (Red)** behaves drastically differently near the barrier compared to BS.
                This is due to the **Vega-Spot Cross Gamma**: As Spot drops, Vol rises (Skew), which makes the option price drop even faster (or recover slower), changing the hedge ratio dynamically.
                """)
                
            except Exception as e:
                st.error(f"An error occurred during calculation: {e}")
                import traceback
                st.text(traceback.format_exc())

    # ---------------------------------------------------------
    # Module 4: Smile Calibration
    # ---------------------------------------------------------
    elif nav_option == "4. Smile Calibration":
        st.subheader("SVI Fit Inspection")
        if not surface.svi_params:
            st.warning("No calibration data available.")
        else:
            available_expiries = sorted(surface.svi_params.keys())
            selected_T = st.selectbox("Select Expiration (Years)", available_expiries)
            
            if selected_T:
                slice_data = surface.raw_data[surface.raw_data['T'] == selected_T]
                
                if slice_data.empty:
                    st.warning("No raw data found for this slice (cleaning filter?).")
                else:
                    k_market = np.log(slice_data['moneyness'])
                    vol_market = slice_data['impliedVolatility']
                    
                    # Generate SVI Curve
                    k_grid = np.linspace(k_market.min()-0.1, k_market.max()+0.1, 100)
                    
                    # Use the updated get_implied_vol which expects log-spot-moneyness
                    vol_model = [surface.get_implied_vol(k, selected_T) for k in k_grid]
                    m_grid = np.exp(k_grid) # Back to Strike/Spot
                    
                    fig_2d = go.Figure()
                    fig_2d.add_trace(go.Scatter(x=slice_data['moneyness'], y=vol_market, mode='markers', name='Market Data', marker=dict(color='red', size=8, symbol='x')))
                    fig_2d.add_trace(go.Scatter(x=m_grid, y=vol_model, mode='lines', name='SVI Model', line=dict(color='blue', width=3)))
                    
                    fig_2d.update_layout(
                        title=f"SVI Calibration: {ticker} (T={selected_T:.2f} Years)", 
                        xaxis_title="Moneyness (Strike / Spot)", 
                        yaxis_title="Implied Volatility",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)
                    
                    # Show Params
                    params = surface.svi_params[selected_T]
                    st.json(params)

    # ---------------------------------------------------------
    # Module 5: Raw Data
    # ---------------------------------------------------------
    elif nav_option == "5. Raw Data Inspector":
        st.subheader("Cleaned Market Data")
        st.dataframe(surface.raw_data)

else:
    st.info("👈 Click **'Build Surface'** in the sidebar to start.")