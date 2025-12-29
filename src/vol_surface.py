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
        
        Args:
            S_t: Current asset price
            t: Current time to maturity
            
        Returns:
            Local Volatility (sigma_loc)
        """
        if not self.svi_params:
            raise ValueError("Surface not built.")

        # 1. Convert Spot to Log-Moneyness k = ln(K/S_0)
        # Note: In Dupire, we usually fix Strike K and vary T. 
        # Here we approximate: k corresponds to the log-moneyness of the strike relative to current spot.
        # Let's use the standard representation: w(k, t) where k = log(K/Spot)
        
        # We need derivatives at (k, t). Let's assume K = S_t (ATM) for the path simulation,
        # but technically Dupire uses K.
        # k = ln(K / S_0).
        # For the pricer, we need sigma_loc(S_t, t). 
        # Relation: k = ln(S_t / S_0) (if we view it as spot evolution)
        
        k = np.log(S_t / self.spot_price)
        
        # Small perturbations for Finite Difference
        dt = 0.001
        dk = 0.01

        # 2. Get Total Variance w(k, t)
        # w = sigma_imp^2 * t
        def get_w(k_val, t_val):
            # Clamp t to avoid looking up T<0
            if t_val < 1e-4: t_val = 1e-4
            vol = self.get_implied_vol(k_val, t_val)
            return (vol ** 2) * t_val

        w = get_w(k, t)

        # 3. Calculate Derivatives (Central Difference)
        # dw/dt (Time slope)
        dw_dt = (get_w(k, t + dt) - get_w(k, t - dt)) / (2 * dt)
        
        # dw/dk (Strike slope)
        dw_dk = (get_w(k + dk, t) - get_w(k - dk, t)) / (2 * dk)
        
        # d2w/dk2 (Strike curvature)
        d2w_dk2 = (get_w(k + dk, t) - 2*w + get_w(k - dk, t)) / (dk ** 2)

        # 4. Dupire Formula
        # numerator = dw/dt
        # denominator = 1 - (k/w)*dw/dk + 0.25*(-0.25 - 1/w + (k/w)^2)*(dw/dk)^2 + 0.5*d2w/dk2
        
        # Safety checks to avoid division by zero
        if w < 1e-6: w = 1e-6
        
        term1 = 1 - (k / w) * dw_dk
        term2 = 0.25 * (-0.25 - (1/w) + (k/w)**2) * (dw_dk**2)
        term3 = 0.5 * d2w_dk2
        
        denominator = term1 + term2 + term3
        
        # Sanity check for negative variance (Arbitrage violation)
        if denominator < 1e-6 or dw_dt < 0:
            # Fallback to Implied Vol if Dupire fails locally
            return np.sqrt(w / t) if t > 0 else 0.0
            
        var_loc = dw_dt / denominator
        
        if var_loc < 0: 
            return np.sqrt(w / t) # Fallback
            
        return np.sqrt(var_loc)

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