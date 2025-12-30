import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple

from src.svi_model import SVIModel
from src.data_loader import DataLoader

# Configure logger
logger = logging.getLogger(__name__)

class VolatilitySurface:
    """
    Constructs the Volatility Surface and extracts Local Volatility using Dupire's Formula.
    """

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker
        self.data_loader = DataLoader(ticker)
        self.svi_params = {} # {expiration_time: svi_params_dict}
        self.spot_price = None
        self.raw_data = None
        
        self.moneyness_grid = np.linspace(0.8, 1.2, 50)
        self.time_grid = None 

    def build(self):
        """
        Main pipeline: Fetches data, calibrates SVI for each slice.
        """
        self.raw_data = self.data_loader.fetch_options_chain()
        self.spot_price = self.data_loader.spot_price
        
        expirations = sorted(self.raw_data['expirationDate'].unique())
        self.time_grid = []

        logger.info(f"Building Vol Surface for {self.ticker} across {len(expirations)} expiries.")

        for exp in expirations:
            df_slice = self.raw_data[self.raw_data['expirationDate'] == exp]
            if len(df_slice) < 5: continue

            T = df_slice['T'].iloc[0]
            k_data = np.log(df_slice['moneyness']).values
            w_data = ((df_slice['impliedVolatility'] ** 2) * T).values
            
            svi = SVIModel()
            params = svi.calibrate(k_data, w_data)
            
            if params and params['success']:
                self.svi_params[T] = params
                self.time_grid.append(T)
            else:
                logger.warning(f"Calibration failed for expiry T={T:.4f}")

        logger.info(f"Surface built. Calibrated {len(self.svi_params)} slices.")

    def get_implied_vol(self, k: float, T: float) -> float:
        """
        Returns Implied Volatility for log-moneyness k and time T.
        Interpolates Total Variance linearly in time.
        """
        if not self.svi_params:
            raise ValueError("Surface not built.")
            
        sorted_times = sorted(self.svi_params.keys())
        T = max(sorted_times[0], min(T, sorted_times[-1]))
        
        if T in self.svi_params:
            return self._get_slice_vol(k, T)

        # Linear Interpolation of Total Variance (w)
        idx = np.searchsorted(sorted_times, T)
        t_lower = sorted_times[idx - 1]
        t_upper = sorted_times[idx]
        
        w_lower = (self._get_slice_vol(k, t_lower) ** 2) * t_lower
        w_upper = (self._get_slice_vol(k, t_upper) ** 2) * t_upper
        
        ratio = (T - t_lower) / (t_upper - t_lower)
        w_target = w_lower + ratio * (w_upper - w_lower)
        
        if w_target < 0: return 0.01
        return np.sqrt(w_target / T)

    def _get_slice_vol(self, k: float, T: float) -> float:
        params = self.svi_params[T]
        svi = SVIModel()
        svi.params = [params['a'], params['b'], params['rho'], params['m'], params['sigma']]
        return svi.get_vol(k, T)

    def get_local_vol(self, S_t: float, t: float) -> float:
        """
        Calculates Local Volatility using Dupire's Formula via Finite Differences.
        Includes safeguards against numerical instability.
        """
        if not self.svi_params:
            raise ValueError("Surface not built.")

        # Cap time to avoid division by zero
        if t < 0.001: t = 0.001
            
        # Current Implied Vol (Fallback)
        k = np.log(S_t / self.spot_price)
        imp_vol = self.get_implied_vol(k, t)
        
        # Finite Difference Steps
        dt = 0.005 # Increased slightly for stability
        dk = 0.02

        def get_w(k_val, t_val):
            t_val = max(1e-4, t_val)
            vol = self.get_implied_vol(k_val, t_val)
            return (vol ** 2) * t_val

        w = get_w(k, t)

        # Derivatives
        dw_dt = (get_w(k, t + dt) - get_w(k, t - dt)) / (2 * dt)
        dw_dk = (get_w(k + dk, t) - get_w(k - dk, t)) / (2 * dk)
        d2w_dk2 = (get_w(k + dk, t) - 2*w + get_w(k - dk, t)) / (dk ** 2)

        # Dupire Denominator
        # Formula: 1 - (k/w)*dw/dk + 0.25*(-0.25 - 1/w + (k/w)^2)*(dw/dk)^2 + 0.5*d2w/dk2
        
        # Safety: w cannot be 0
        if w < 1e-6: w = 1e-6
        
        term1 = 1 - (k / w) * dw_dk
        term2 = 0.25 * (-0.25 - (1/w) + (k/w)**2) * (dw_dk**2)
        term3 = 0.5 * d2w_dk2
        
        denominator = term1 + term2 + term3
        
        # 1. Negative time slope implies Calendar Arbitrage (Var decreases with time)
        if dw_dt < 0:
            return imp_vol
            
        # 2. Denominator close to zero or negative implies Static Arbitrage
        if denominator < 1e-4:
            return imp_vol
            
        var_loc = dw_dt / denominator
        
        if var_loc < 0:
            return imp_vol
            
        vol_loc = np.sqrt(var_loc)
        
        # 3. Cap extreme spikes (e.g., > 300% vol is likely numerical noise for SPY)
        if vol_loc > 3.0: 
            return imp_vol
            
        return vol_loc

    def get_mesh_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 3D coordinates for BOTH Implied and Local Volatility.
        Returns: X, Y, Z_implied, Z_local
        """
        if not self.time_grid:
            self.build()
            
        valid_times = sorted(self.svi_params.keys())
        # Filter times to avoid T=0 issues
        valid_times = [t for t in valid_times if t > 0.05]
        
        X, Y = np.meshgrid(self.moneyness_grid, valid_times)
        Z_imp = np.zeros_like(X)
        Z_loc = np.zeros_like(X)
        
        for i, t in enumerate(valid_times):
            for j, m_val in enumerate(self.moneyness_grid):
                k = np.log(m_val)
                # Spot price implied by moneyness: S = S0 * exp(-k) ?? 
                # Actually simpler: moneyness = K/S0 -> K = m_val * S0
                # Dupire takes S_t. If we are plotting surface over Moneyness K/S0,
                # we calculate Local Vol assuming the Spot moves to that Strike level.
                S_equivalent = self.spot_price * m_val
                
                Z_imp[i, j] = self.get_implied_vol(k, t)
                Z_loc[i, j] = self.get_local_vol(S_equivalent, t)
                
        return X, Y, Z_imp, Z_loc

# --- Test Block ---
if __name__ == "__main__":
    surface = VolatilitySurface("SPY")
    surface.build()