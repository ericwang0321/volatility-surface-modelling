import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from scipy.interpolate import RegularGridInterpolator

from src.svi_model import SVIModel
from src.data_loader import DataLoader
from src.rates import RateProvider

# Configure logger
logger = logging.getLogger(__name__)

class VolatilitySurface:
    """
    Constructs the Volatility Surface and extracts Local Volatility.
    
    MAJOR UPDATE:
    - Now uses **Log-Forward Moneyness** (y = log(K/F)) instead of Spot Moneyness.
    - This simplifies Dupire's formula (drift term cancels out) and aligns with industry standards.
    """

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker
        self.data_loader = DataLoader(ticker)
        self.rate_provider = RateProvider() 
        self.svi_params = {} 
        self.spot_price = None
        self.dividend_yield = 0.0
        self.raw_data = None
        
        # Grid settings for surface generation
        # NOTE: Grid now represents Log-Forward Moneyness, centered around 0 (ATM)
        # e.g., -0.2 to +0.2 roughly corresponds to 80% to 120% Forward Moneyness
        self.moneyness_grid = np.linspace(np.log(0.8), np.log(1.2), 50) 
        self.time_grid = None 

        # Interpolator for Local Volatility
        self.interpolator = None
        self.interp_s_grid = None
        self.interp_t_grid = None

    def build(self):
        """
        Main pipeline: Fetches data, converts to Forward coordinates, calibrates SVI.
        """
        # Fetch Data
        self.raw_data = self.data_loader.fetch_options_chain()
        self.spot_price = self.data_loader.spot_price
        self.dividend_yield = self.data_loader.fetch_dividend_yield()
        
        # Identify unique expirations
        expirations = sorted(self.raw_data['expirationDate'].unique())
        self.time_grid = []

        logger.info(f"Building Vol Surface for {self.ticker} (Log-Forward) across {len(expirations)} expiries.")

        prev_w_atm = -1.0

        for exp in expirations:
            df_slice = self.raw_data[self.raw_data['expirationDate'] == exp]
            if len(df_slice) < 5: 
                continue

            T = df_slice['T'].iloc[0]
            if T < 0.002: continue

            # --- COORDINATE TRANSFORMATION ---
            # 1. Get Rates & Dividend
            r = self.rate_provider.get_risk_free_rate(T)
            q = self.dividend_yield
            
            # 2. Calculate Log-Spot Moneyness k_spot = log(K/S)
            k_spot = np.log(df_slice['moneyness']).values
            
            # 3. Convert to Log-Forward Moneyness y = log(K/F)
            # F = S * exp((r-q)T)  =>  log(K/F) = log(K/S) - (r-q)T
            y_data = k_spot - (r - q) * T
            
            w_data = ((df_slice['impliedVolatility'] ** 2) * T).values
            
            # Calibrate SVI on y (Log-Forward)
            svi = SVIModel()
            params = svi.calibrate(y_data, w_data)
            
            if params and params['success']:
                # Calendar Arb Check (at ATM Forward, y=0)
                current_w_atm = svi.raw_svi_formula(0, params['a'], params['b'], params['rho'], params['m'], params['sigma'])
                
                if prev_w_atm > 0 and current_w_atm < prev_w_atm:
                    logger.warning(f"Potential Calendar Arbitrage at T={T:.4f}. Variance decreased. Skipping.")
                    continue
                
                self.svi_params[T] = params
                self.time_grid.append(T)
                prev_w_atm = current_w_atm

        logger.info(f"Surface built. Calibrated {len(self.svi_params)} slices.")

    def build_local_vol_interpolator(self, s_steps: int = 100, t_steps: int = 50):
        """
        Pre-computes Local Volatility on a dense grid for fast lookup.
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

    def get_implied_vol(self, k_spot: float, T: float) -> float:
        """
        Returns Implied Volatility.
        NOTE: Input `k_spot` is still log(Strike/Spot) for compatibility with external callers.
        We convert it to log-forward `y` internally.
        """
        if not self.svi_params:
            raise ValueError("Surface not built.")
            
        sorted_times = sorted(self.svi_params.keys())
        T_clamped = max(sorted_times[0], min(T, sorted_times[-1]))
        
        # Convert Spot Moneyness -> Forward Moneyness
        r = self.rate_provider.get_risk_free_rate(T_clamped)
        q = self.dividend_yield
        y = k_spot - (r - q) * T_clamped

        # Exact match
        if T_clamped in self.svi_params:
            return self._get_slice_vol(y, T_clamped)

        # Interpolation
        idx = np.searchsorted(sorted_times, T_clamped)
        t_lower = sorted_times[idx - 1]
        t_upper = sorted_times[idx]
        
        # We need to re-calculate y for lower/upper T because forward price changes with T!
        r_lower = self.rate_provider.get_risk_free_rate(t_lower)
        y_lower = k_spot - (r_lower - q) * t_lower
        
        r_upper = self.rate_provider.get_risk_free_rate(t_upper)
        y_upper = k_spot - (r_upper - q) * t_upper
        
        w_lower = (self._get_slice_vol(y_lower, t_lower) ** 2) * t_lower
        w_upper = (self._get_slice_vol(y_upper, t_upper) ** 2) * t_upper
        
        # Calendar Arb clamp
        if w_upper < w_lower: w_upper = w_lower

        ratio = (T_clamped - t_lower) / (t_upper - t_lower)
        w_target = w_lower + ratio * (w_upper - w_lower)
        
        if w_target < 1e-6: w_target = 1e-6
        return np.sqrt(w_target / T_clamped)

    def _get_slice_vol(self, y: float, T: float) -> float:
        """y is Log-Forward Moneyness here."""
        params = self.svi_params[T]
        return SVIModel.raw_svi_formula_vol(y, params['a'], params['b'], params['rho'], params['m'], params['sigma'], T)

    def get_local_vol(self, S_t: float, t: float) -> float:
        """
        Calculates Local Volatility using Dupire's Formula in Log-Forward coordinates.
        
        Formula (Log-Forward y):
        sigma_loc^2 = (dw/dT) / (1 - y/w * dw/dy + 0.25(-0.25 - 1/w + y^2/w^2)(dw/dy)^2 + 0.5 d2w/dy2)
        
        Key Advantage: The drift term (r-q) * dw/dk is ABSORBED into the coordinate system.
        Numerator is simply dw/dT.
        """
        if t < 0.002: t = 0.002
        if S_t <= 1e-3: S_t = 1e-3
        
        # 1. Coordinate Prep
        r = self.rate_provider.get_risk_free_rate(t)
        q = self.dividend_yield
        
        # Initial Spot S0 (Surface Base)
        S0 = self.spot_price 
        
        # We need Local Vol at Strike K = S_t
        K_equiv = S_t
        
        # Log-Forward Moneyness y = log(K / F_t)
        # F_t = S0 * exp((r-q)t)
        # y = log(K / S0) - (r-q)t
        k_spot = np.log(K_equiv / S0)
        y = k_spot - (r - q) * t
        
        # 2. Helper for Total Variance w(y, t)
        def get_w(y_in, t_in):
            t_in = max(1e-4, t_in)
            # Need to handle y -> vol conversion carefully during finite difference
            # SVI is calibrated on y, so we can call _get_slice_vol directly IF t_in is an exact slice.
            # But t_in is continuous (t + dt). So we use get_implied_vol logic but tailored for y input?
            # Actually, our get_implied_vol takes k_spot.
            # Let's reverse engineer k_spot for the helper to reuse logic.
            
            # y_in = k_spot_in - (r-q)t_in
            # k_spot_in = y_in + (r-q)t_in
            # This keeps the abstraction consistent.
            r_in = self.rate_provider.get_risk_free_rate(t_in)
            k_spot_temp = y_in + (r_in - q) * t_in
            
            vol = self.get_implied_vol(k_spot_temp, t_in)
            return (vol ** 2) * t_in

        # 3. Finite Differences
        dt = 0.005
        dy = 0.01 
        
        w = get_w(y, t)
        imp_vol = np.sqrt(w / t)
        
        # Time Derivative (Numerator) - No Drift Term needed!
        w_t_plus = get_w(y, t + dt)
        w_t_minus = get_w(y, t - dt)
        dw_dt = (w_t_plus - w_t_minus) / (2 * dt)
        
        # Moneyness Derivatives (Denominator)
        w_y_plus = get_w(y + dy, t)
        w_y_minus = get_w(y - dy, t)
        dw_dy = (w_y_plus - w_y_minus) / (2 * dy)
        
        d2w_dy2 = (w_y_plus - 2*w + w_y_minus) / (dy**2)
        
        # 4. Dupire Formula (Gatheral's Forward Form)
        numerator = dw_dt 
        
        term1 = 1 - (y / w) * dw_dy
        term2 = 0.25 * (-0.25 - (1/w) + (y/w)**2) * (dw_dy**2)
        term3 = 0.5 * d2w_dy2
        
        denominator = term1 + term2 + term3
        
        # 5. Stability & Clamping
        if numerator < 1e-6: numerator = 1e-6
        if denominator < 1e-6: denominator = 1e-6
            
        var_loc = numerator / denominator
        
        if var_loc < 0: var_loc = 0
        vol_loc = np.sqrt(var_loc)
        
        # Ratio Cap
        if vol_loc > imp_vol * 2.5: vol_loc = imp_vol * 2.5
        if vol_loc < imp_vol * 0.2: vol_loc = imp_vol * 0.2
            
        return vol_loc

    def get_mesh_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates grid data for plotting.
        Converts internal Log-Forward y back to Spot Moneyness M = K/S for intuitive plotting.
        """
        if not self.time_grid: self.build()
        valid_times = sorted([t for t in self.svi_params.keys() if t > 0.05])
        
        # Plotting grid in Spot Moneyness (e.g. 0.8 to 1.2)
        spot_moneyness_plot = np.linspace(0.8, 1.2, 50)
        
        X, Y = np.meshgrid(spot_moneyness_plot, valid_times)
        Z_imp = np.zeros_like(X)
        Z_loc = np.zeros_like(X)
        
        for i, t in enumerate(valid_times):
            for j, m_val in enumerate(spot_moneyness_plot):
                # k_spot = log(m_val)
                k_spot = np.log(m_val)
                S_eq = self.spot_price * m_val
                
                Z_imp[i, j] = self.get_implied_vol(k_spot, t)
                Z_loc[i, j] = self.get_local_vol(S_eq, t)
        
        return X, Y, Z_imp, Z_loc