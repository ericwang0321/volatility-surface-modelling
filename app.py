import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from src.vol_surface import VolatilitySurface

st.set_page_config(page_title="Quant Volatility Surface", layout="wide")

# Title and Description
st.title("📈 Industrial Volatility Surface & Pricing Engine")
st.markdown("""
This dashboard demonstrates an end-to-end quantitative pipeline:
1.  **Data ETL**: Real-time fetching & cleaning of SPY options chain.
2.  **Model Calibration**: Fitting **SVI (Stochastic Volatility Inspired)** model to market smiles.
3.  **Local Volatility**: Extracting **Dupire's Local Volatility** from the implied surface.
""")

# Sidebar settings
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
run_btn = st.sidebar.button("Build Surface 🚀")

# Main Logic
if run_btn:
    with st.spinner(f"Fetching data and calibrating models for {ticker}..."):
        try:
            # Initialize and Build
            surface = VolatilitySurface(ticker)
            surface.build()
            
            st.success(f"Successfully calibrated {len(surface.svi_params)} expiration slices!")
            
            # --- Tab Layout ---
            tab1, tab2, tab3 = st.tabs(["📊 3D Surfaces (Imp vs Loc)", "😊 Volatility Smile", "📋 Raw Data"])
            
            # --- Tab 1: 3D Surfaces (Implied & Local) ---
            with tab1:
                # Get all grid data: X, Y, Implied Z, Local Z
                X, Y, Z_imp, Z_loc = surface.get_mesh_grid()
                
                col1, col2 = st.columns(2)
                
                # Plot 1: Implied Volatility Surface
                with col1:
                    st.subheader("1. Implied Volatility Surface")
                    fig_imp = go.Figure(data=[go.Surface(
                        z=Z_imp, x=X, y=Y,
                        colorscale='Viridis',
                        colorbar_title='Implied Vol'
                    )])
                    
                    fig_imp.update_layout(
                        title=f'{ticker} Implied Vol (Market Expectations)',
                        scene=dict(
                            xaxis_title='Moneyness (K/S)',
                            yaxis_title='Time (Years)',
                            zaxis_title='Vol'
                        ),
                        width=500, height=500,
                        margin=dict(l=10, r=10, b=10, t=40)
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                # Plot 2: Local Volatility Surface
                with col2:
                    st.subheader("2. Local Volatility Surface (Dupire)")
                    fig_loc = go.Figure(data=[go.Surface(
                        z=Z_loc, x=X, y=Y,
                        colorscale='Turbo', # Different color scale for distinction
                        colorbar_title='Local Vol'
                    )])
                    
                    fig_loc.update_layout(
                        title=f'{ticker} Local Vol (Pricing Input)',
                        scene=dict(
                            xaxis_title='Spot / Strike',
                            yaxis_title='Time (Years)',
                            zaxis_title='Local Vol'
                        ),
                        width=500, height=500,
                        margin=dict(l=10, r=10, b=10, t=40)
                    )
                    st.plotly_chart(fig_loc, use_container_width=True)
                
                st.info("""
                💡 **Quant Insight:** * **Left (Implied):** Represents the market's *average* volatility expectation over the life of the option. It is smooth.
                * **Right (Local):** Represents the *instantaneous* volatility at a specific Spot and Time. Notice how it is much "spikier" and steeper? 
                This is because Local Volatility must amplify the skew to explain the market prices mathematically. It is the "true" volatility input for Monte Carlo pricing.
                """)

            # --- Tab 2: 2D Smile Calibration ---
            with tab2:
                st.subheader("SVI Calibration Check")
                
                # Let user choose an expiry to inspect
                available_expiries = sorted(surface.svi_params.keys())
                
                if not available_expiries:
                    st.warning("No expiries calibrated.")
                else:
                    selected_T = st.selectbox("Select Expiration (Years)", available_expiries)
                    
                    if selected_T:
                        # Get Raw Data for this slice
                        slice_data = surface.raw_data[surface.raw_data['T'] == selected_T]
                        
                        # Prepare Plot Data
                        k_market = np.log(slice_data['moneyness'])
                        vol_market = slice_data['impliedVolatility']
                        
                        # Generate Model Curve
                        # Extend grid slightly beyond data to show curvature
                        k_min, k_max = k_market.min(), k_market.max()
                        k_grid = np.linspace(k_min - 0.1, k_max + 0.1, 100)
                        
                        vol_model = [surface.get_implied_vol(k, selected_T) for k in k_grid]
                        
                        # Convert k back to Moneyness for plotting (x-axis)
                        m_grid = np.exp(k_grid)
                        
                        # Plot
                        fig_2d = go.Figure()
                        fig_2d.add_trace(go.Scatter(
                            x=slice_data['moneyness'], y=vol_market,
                            mode='markers', name='Market Data',
                            marker=dict(color='red', size=8, symbol='x')
                        ))
                        fig_2d.add_trace(go.Scatter(
                            x=m_grid, y=vol_model,
                            mode='lines', name='SVI Model',
                            line=dict(color='blue', width=3)
                        ))
                        
                        fig_2d.update_layout(
                            title=f"Volatility Smile at T={selected_T:.4f}",
                            xaxis_title="Moneyness (Strike/Spot)",
                            yaxis_title="Implied Volatility",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_2d, use_container_width=True)

            # --- Tab 3: Data Inspection ---
            with tab3:
                st.subheader("Cleaned Market Data")
                st.dataframe(surface.raw_data)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            # Optional: Print traceback for debugging
            import traceback
            st.text(traceback.format_exc())

else:
    st.info("👈 Click **'Build Surface'** in the sidebar to start the pipeline.")