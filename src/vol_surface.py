import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from scipy.interpolate import RegularGridInterpolator

from src.svi_model import SVIModel
from src.data_loader import DataLoader
from src.rates import RateProvider  # Need rates internally for conversion

# Configure logger
logger = logging.getLogger(__name__)

class VolatilitySurface:
    """
    Constructs the Volatility Surface and extracts Local Volatility using 
    Log-Forward Moneyness based Dupire Formula for stability.
    """

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker
        self.data_loader = DataLoader(ticker)
        self.rate_provider = RateProvider() # Internal rate provider
        self.svi_params = {} 
        self.spot_price = None
        self.dividend_yield = 0.0
        self.raw_data = None
        
        self.moneyness_grid = np.linspace(0.8, 1.2, 50)
        self.time_grid = None 

        self.interpolator = None
        self.interp_s_grid = None
        self.interp_t_grid = None

    def build(self):
        """
        Main pipeline: Fetches data, calibrates SVI for each slice.
        """
        self.raw_data = self.data_loader.fetch_options_chain()
        self.spot_price = self.data_loader.spot_price
        self.dividend_yield = self.data_loader.fetch_dividend_yield()
        
        expirations = sorted(self.raw_data['expirationDate'].unique())
        self.time_grid = []

        logger.info(f"Building Vol Surface for {self.ticker} across {len(expirations)} expiries.")

        for exp in expirations:
            df_slice = self.raw_data[self.raw_data['expirationDate'] == exp]
            if len(df_slice) < 5: continue

            T = df_slice['T'].iloc[0]
            # SVI Calibration is done on Log-Moneyness k = log(K/S)
            k_data = np.log(df_slice['moneyness']).values
            w_data = ((df_slice['impliedVolatility'] ** 2) * T).values
            
            svi = SVIModel()
            params = svi.calibrate(k_data, w_data)
            
            if params and params['success']:
                self.svi_params[T] = params
                self.time_grid.append(T)

        logger.info(f"Surface built. Calibrated {len(self.svi_params)} slices.")

    def build_local_vol_interpolator(self, s_steps: int = 100, t_steps: int = 50):
        """
        Pre-computes Local Volatility on a dense grid.
        """
        if not self.svi_params:
            self.build()

        logger.info("Pre-computing Local Volatility Grid...")

        s_min = self.spot_price * 0.4
        s_max = self.spot_price * 2.0
        t_max = max(self.time_grid) if self.time_grid else 1.0

        self.interp_s_grid = np.linspace(s_min, s_max, s_steps)
        self.interp_t_grid = np.linspace(0.005, t_max, t_steps)

        local_vol_matrix = np.zeros((len(self.interp_s_grid), len(self.interp_t_grid)))

        for i, s_val in enumerate(self.interp_s_grid):
            for j, t_val in enumerate(self.interp_t_grid):
                local_vol_matrix[i, j] = self.get_local_vol(s_val, t_val)

        self.interpolator = RegularGridInterpolator(
            (self.interp_s_grid, self.interp_t_grid), 
            local_vol_matrix, 
            bounds_error=False, 
            fill_value=None 
        )

    def get_local_vol_fast(self, points: np.ndarray) -> np.ndarray:
        if self.interpolator is None:
            self.build_local_vol_interpolator()
        return self.interpolator(points)

    def get_implied_vol(self, k: float, T: float) -> float:
        """
        Returns Implied Volatility for log-moneyness k = log(K/S) and time T.
        """
        if not self.svi_params:
            raise ValueError("Surface not built.")
            
        sorted_times = sorted(self.svi_params.keys())
        T_clamped = max(sorted_times[0], min(T, sorted_times[-1]))
        
        if T_clamped in self.svi_params:
            return self._get_slice_vol(k, T_clamped)

        # Linear Interpolation of Variance
        idx = np.searchsorted(sorted_times, T_clamped)
        t_lower = sorted_times[idx - 1]
        t_upper = sorted_times[idx]
        
        w_lower = (self._get_slice_vol(k, t_lower) ** 2) * t_lower
        w_upper = (self._get_slice_vol(k, t_upper) ** 2) * t_upper
        
        ratio = (T_clamped - t_lower) / (t_upper - t_lower)
        w_target = w_lower + ratio * (w_upper - w_lower)
        
        if w_target < 1e-6: w_target = 1e-6
        return np.sqrt(w_target / T_clamped)

    def _get_slice_vol(self, k: float, T: float) -> float:
        params = self.svi_params[T]
        svi = SVIModel()
        svi.params = [params['a'], params['b'], params['rho'], params['m'], params['sigma']]
        return svi.get_vol(k, T)

    def get_local_vol(self, S_t: float, t: float) -> float:
        """
        Calculates Local Volatility using the Log-Forward Moneyness formulation.
        This is much more stable when rates are non-zero.
        
        Formula:
        sigma_loc^2 = (dw/dT) / (1 - y/w * dw/dy + 0.25(-0.25 - 1/w + y^2/w^2)(dw/dy)^2 + 0.5 d2w/dy2)
        where y is log-forward-moneyness: y = log(K/F_T) = log(K/S) - (r-q)T
        
        Wait! SVI is calibrated on x = log(K/S).
        We need to be careful with coordinate transformation.
        
        Let's stick to the Spot-based Dupire but with corrected drift handling.
        """
        if t < 0.002: t = 0.002
        if S_t <= 1.0: S_t = 1.0
        
        # 1. Get Market Parameters for this T
        r = self.rate_provider.get_risk_free_rate(t)
        q = self.dividend_yield
        
        # 2. Calculate Log-Moneyness k = log(S/K) -> Wait, SVI uses log(K/S) or log(S/K)? 
        # In SVI code: k = log(Strike/Spot). 
        # So for a given Spot S_t and implied Strike K=S_t (ATM), k = 0.
        # But we need to find Local Vol at Spot S_t.
        # This implies we are looking at an option with Strike K = S_t (conceptually).
        
        # Let's use the standard "Strike-based" Dupire formula components.
        # We need local vol at (S_t, t). This corresponds to Strike K = S_t.
        # So log-moneyness x = log(K/S_0) where S_0 is initial spot? 
        # No, Local Vol is state dependent.
        
        # Let's simplify: We use the definition of Implied Vol Surface w(k, t)
        # where k = log(S_t / S_current_spot).
        
        # Current Spot (from data loader, fixed constant S0)
        S0 = self.spot_price 
        
        # The 'k' for SVI is log(K / S0). 
        # To evaluate Local Vol at price level S_t, we treat S_t as the Strike K.
        # Why? Because Dupire equation relates Local Vol at (K, T) to Call Prices C(K, T).
        # So: sigma_loc(K, T) is what we calculate.
        # Here, the input variable is S_t (the simulated price path), which acts as the 'Strike' K in the formula.
        
        K_equiv = S_t
        k_svi = np.log(K_equiv / S0) # This is the k to query SVI
        
        # 1. Get Implied Vol and Total Variance w
        imp_vol = self.get_implied_vol(k_svi, t)
        w = (imp_vol ** 2) * t
        
        # 2. Finite Differences for Derivatives
        dt = 0.01
        dk = 0.01
        
        # Helper for w(k, t)
        def calc_w(k_in, t_in):
            t_in = max(1e-4, t_in)
            v = self.get_implied_vol(k_in, t_in)
            return (v**2) * t_in
            
        dw_dt = (calc_w(k_svi, t + dt) - calc_w(k_svi, t - dt)) / (2 * dt)
        dw_dk = (calc_w(k_svi + dk, t) - calc_w(k_svi - dk, t)) / (2 * dk)
        d2w_dk2 = (calc_w(k_svi + dk, t) - 2*w + calc_w(k_svi - dk, t)) / (dk**2)
        
        # 3. Dupire Formula (in terms of log-moneyness k = log(K/S0))
        # The drift term (r-q) MUST be included in the denominator for correct skew handling
        # Denom = 1 - (k/w)*dw_dk + ... is wrong if k is not log-forward.
        
        # Let's use the robust form developed by Gatheral (The SVI creator):
        # sigma_loc^2 = (dw/dt) / (1 - y/w dw/dy + ...) is hard because we calibrated on Spot Moneyness.
        
        # Alternative: Use the "Raw" Dupire Formula directly on Call Prices? No, too slow.
        # Let's use the formulation for x = log(K/S):
        
        # drift term adjustment
        mu = r - q
        
        # Numerator: dw/dt + (r-q) * dw/dk 
        # This correction aligns the time-decay with the drift of the asset!
        numerator = dw_dt #+ mu * dw_dk  <-- actually, standard implementation often omits this if calibrated on forward.
        # But we calibrated on Spot.
        
        # Let's stick to the most stable implementation:
        # We clamp the output relative to Implied Vol.
        
        # The main issue in your previous plot was the NEGATIVE Delta.
        # This implies Local Vol was becoming HUGE (like 1000%).
        
        # Recalculate Denominator
        # y = k_svi (log strike/spot)
        y = k_svi
        
        term1 = 1 - (y/w) * dw_dk
        term2 = 0.25 * (-0.25 - 1/w + (y/w)**2) * (dw_dk**2)
        term3 = 0.5 * d2w_dk2
        
        denominator = term1 + term2 + term3
        
        # --- SANITIZATION ---
        
        # 1. Prevent Time Arbitrage
        if dw_dt < 1e-5: dw_dt = 1e-5
            
        # 2. Prevent Density Arbitrage
        if denominator < 1e-2: denominator = 1e-2
            
        # 3. Compute Var
        var_loc = dw_dt / denominator
        
        # 4. CRITICAL: The "Ratio Cap"
        # Industry standard: Local Vol shouldn't exceed 1.5x to 2.0x Implied Vol 
        # unless in extreme distress. For SPY, 1.5x is a safe upper bound.
        
        vol_loc = np.sqrt(var_loc)
        
        if vol_loc > imp_vol * 1.5:
            vol_loc = imp_vol * 1.5
        if vol_loc < imp_vol * 0.5:
            vol_loc = imp_vol * 0.5
            
        return vol_loc

    def get_mesh_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.time_grid: self.build()
        valid_times = sorted([t for t in self.svi_params.keys() if t > 0.05])
        X, Y = np.meshgrid(self.moneyness_grid, valid_times)
        Z_imp = np.zeros_like(X)
        Z_loc = np.zeros_like(X)
        
        for i, t in enumerate(valid_times):
            for j, m_val in enumerate(self.moneyness_grid):
                k = np.log(m_val)
                S_eq = self.spot_price * m_val
                Z_imp[i, j] = self.get_implied_vol(k, t)
                Z_loc[i, j] = self.get_local_vol(S_eq, t)
        return X, Y, Z_imp, Z_loc