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
3.  **Surface Construction**: Interpolating in time to build a dense volatility surface.
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
            tab1, tab2, tab3 = st.tabs(["📊 3D Surface", "😊 Volatility Smile", "📋 Raw Data"])
            
            # --- Tab 1: 3D Surface ---
            with tab1:
                st.subheader("Implied Volatility Surface")
                X, Y, Z = surface.get_mesh_grid()
                
                # Plotly 3D Surface
                fig_3d = go.Figure(data=[go.Surface(
                    z=Z, x=X, y=Y,
                    colorscale='Viridis',
                    colorbar_title='Implied Vol'
                )])
                
                fig_3d.update_layout(
                    title=f'{ticker} Implied Volatility Surface',
                    scene=dict(
                        xaxis_title='Moneyness (K/S)',
                        yaxis_title='Time to Maturity (Years)',
                        zaxis_title='Implied Volatility'
                    ),
                    width=900, height=600,
                    margin=dict(l=65, r=50, b=65, t=90)
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.info("💡 **Quant Insight:** Notice how the surface skews upwards for lower moneyness (Out-of-the-Money Puts). This reflects the market's fear of crash risk (The Skew).")

            # --- Tab 2: 2D Smile Calibration ---
            with tab2:
                st.subheader("SVI Calibration Check")
                
                # Let user choose an expiry to inspect
                available_expiries = sorted(surface.svi_params.keys())
                selected_T = st.selectbox("Select Expiration (Years)", available_expiries)
                
                if selected_T:
                    # Get Raw Data for this slice
                    slice_data = surface.raw_data[surface.raw_data['T'] == selected_T]
                    k_market = np.log(slice_data['moneyness'])
                    vol_market = slice_data['impliedVolatility']
                    
                    # Generate Model Curve
                    k_grid = np.linspace(k_market.min()-0.1, k_market.max()+0.1, 100)
                    vol_model = [surface.get_implied_vol(k, selected_T) for k in k_grid]
                    
                    # Convert k back to Moneyness for plotting
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
                        title=f"Volatility Smile at T={selected_T:.2f}",
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

else:
    st.info("👈 Click **'Build Surface'** in the sidebar to start the pipeline.")