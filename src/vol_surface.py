import numpy as np
import pandas as pd
import logging
from scipy.interpolate import interp1d
from typing import Dict, Tuple

from src.svi_model import SVIModel
from src.data_loader import DataLoader

# Configure logger
logger = logging.getLogger(__name__)

class VolatilitySurface:
    """
    Constructs the Volatility Surface by calibrating SVI slices 
    for each expiration and interpolating in time.
    """

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker
        self.data_loader = DataLoader(ticker)
        self.svi_params = {} # Stores {expiration_date: svi_params_dict}
        self.spot_price = None
        self.raw_data = None
        
        # Grid settings for the surface
        self.moneyness_grid = np.linspace(0.8, 1.2, 50) # 80% to 120% moneyness
        self.time_grid = None # Will be set dynamically

    def build(self):
        """
        Main pipeline: Fetches data, calibrates SVI for each slice, 
        and prepares the surface.
        """
        # 1. Fetch Data
        self.raw_data = self.data_loader.fetch_options_chain()
        self.spot_price = self.data_loader.spot_price
        
        # 2. Get unique expirations and sort them
        expirations = sorted(self.raw_data['expirationDate'].unique())
        self.time_grid = []

        logger.info(f"Building Vol Surface for {self.ticker} across {len(expirations)} expiries.")

        # 3. Calibrate SVI for each slice
        for exp in expirations:
            # Filter data for this expiry
            df_slice = self.raw_data[self.raw_data['expirationDate'] == exp]
            
            # We need at least 5 points to calibrate 5 SVI params
            if len(df_slice) < 5:
                continue

            T = df_slice['T'].iloc[0]
            
            # Prepare arrays for calibration
            # k = log(K/S) -> .values ensures it's a numpy array
            k_data = np.log(df_slice['moneyness']).values
            
            # w = Total Variance = sigma^2 * T
            # impliedVolatility is a Series, so we convert result to numpy array
            w_data = ((df_slice['impliedVolatility'] ** 2) * T).values
            
            # Calibrate
            svi = SVIModel()
            # FIX: Removed .values here because k_data/w_data are already arrays
            params = svi.calibrate(k_data, w_data)
            
            if params and params['success']:
                self.svi_params[T] = params
                self.time_grid.append(T)
            else:
                logger.warning(f"Calibration failed for expiry T={T:.4f}")

        logger.info(f"Surface built successfully. Calibrated {len(self.svi_params)} slices.")

    def get_implied_vol(self, k: float, T: float) -> float:
        """
        Returns Implied Volatility for arbitrary (k, T).
        Performs linear interpolation on Total Variance in Time dimension.
        """
        if not self.svi_params:
            raise ValueError("Surface not built. Call build() first.")
            
        # 1. Find the two surrounding time slices (T_prev, T_next)
        sorted_times = sorted(self.svi_params.keys())
        
        # Extrapolation protection: Clamp T to min/max range
        T = max(sorted_times[0], min(T, sorted_times[-1]))
        
        # If exact match
        if T in self.svi_params:
            return self._get_slice_vol(k, T)

        # 2. Linear Interpolation in Total Variance Space
        # Locate T in the sorted list
        idx = np.searchsorted(sorted_times, T)
        t_lower = sorted_times[idx - 1]
        t_upper = sorted_times[idx]
        
        # Calculate Total Variance (w) for lower and upper slices
        w_lower = (self._get_slice_vol(k, t_lower) ** 2) * t_lower
        w_upper = (self._get_slice_vol(k, t_upper) ** 2) * t_upper
        
        # Linear Interpolation formula
        ratio = (T - t_lower) / (t_upper - t_lower)
        w_target = w_lower + ratio * (w_upper - w_lower)
        
        # Convert back to Vol
        return np.sqrt(w_target / T)

    def _get_slice_vol(self, k: float, T: float) -> float:
        """Helper to get vol from a specific known SVI slice"""
        params = self.svi_params[T]
        svi = SVIModel()
        svi.params = [params['a'], params['b'], params['rho'], params['m'], params['sigma']]
        return svi.get_vol(k, T)

    def get_mesh_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 3D coordinates (X, Y, Z) for plotting.
        X: Log-Moneyness
        Y: Time to Maturity
        Z: Implied Volatility
        """
        if not self.time_grid:
            self.build()
            
        # Create a dense grid
        # Use only valid calibrated times
        valid_times = sorted(self.svi_params.keys())
        
        X, Y = np.meshgrid(self.moneyness_grid, valid_times)
        Z = np.zeros_like(X)
        
        for i, t in enumerate(valid_times):
            for j, m_val in enumerate(self.moneyness_grid):
                # moneyness_grid is K/S, but model takes log(K/S)
                k = np.log(m_val)
                Z[i, j] = self.get_implied_vol(k, t)
                
        return X, Y, Z

# --- Test Block ---
if __name__ == "__main__":
    surface = VolatilitySurface("SPY")
    surface.build()
    
    # Test a specific point (ATM option, 1 year out)
    vol = surface.get_implied_vol(k=0, T=1.0)
    print(f"\nEstimated ATM Volatility at T=1.0: {vol:.2%}")
    
    # Check surface dimensions
    X, Y, Z = surface.get_mesh_grid()
    print(f"Mesh Grid Generated. Shape: {Z.shape}")