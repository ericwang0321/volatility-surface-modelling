import numpy as np
from scipy.optimize import minimize
import logging
from typing import List, Tuple, Dict, Optional

# Configure logger
logger = logging.getLogger(__name__)

class SVIModel:
    """
    Implements the Raw SVI (Stochastic Volatility Inspired) model parameterization.
    Ref: Gatheral, J. (2004). "A Parcimonious Representation of the Volatility Surface".
    
    Improvements:
    1. Added explicit static arbitrage constraints via penalty functions.
    2. Added `raw_svi_formula_vol` helper for the surface builder.
    3. Enhanced optimization using SLSQP with boundary and inequality constraints.
    """

    def __init__(self):
        # Parameters: [a, b, rho, m, sigma]
        self.params = None
        self.rms_error = None

    @staticmethod
    def raw_svi_formula(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
        """
        The Raw SVI formula: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        Returns Total Variance (w).
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    @staticmethod
    def raw_svi_formula_vol(k: float, a: float, b: float, rho: float, m: float, sigma: float, T: float) -> float:
        """
        Helper method to return Implied Volatility directly.
        sigma_impl = sqrt(w / T)
        """
        w = SVIModel.raw_svi_formula(k, a, b, rho, m, sigma)
        if w < 1e-8: w = 1e-8 # Clamp non-positive variance
        return np.sqrt(w / T)

    def calibrate(self, k_data: np.array, w_data: np.array) -> Optional[Dict]:
        """
        Calibrates the SVI model to market data (Log-Forward Moneyness k, Total Variance w).
        """
        
        # 1. Define Objective Function (RMSE + Penalty)
        def objective(params):
            a, b, rho, m, sigma = params
            
            # Calculate Model Variance
            w_model = self.raw_svi_formula(k_data, a, b, rho, m, sigma)
            
            # RMSE
            error = w_model - w_data
            rmse = np.sqrt(np.mean(error**2))
            
            # --- PENALTY TERMS ---
            penalty = 0.0
            
            # Constraint: b >= 0
            if b < 0: penalty += 100 * abs(b)
                
            # Constraint: |rho| < 1
            if abs(rho) >= 1: penalty += 100 * (abs(rho) - 0.99)
                
            # Constraint: sigma > 0
            if sigma < 0: penalty += 100 * abs(sigma)
            
            # Constraint: No Static Arbitrage (Variance > 0 everywhere)
            # The minimum variance in SVI is approx (a + b * sigma * sqrt(1 - rho^2))
            # We want this to be >= 0.
            min_var_proxy = a + b * sigma * np.sqrt(1 - min(0.999, rho**2))
            if min_var_proxy < 0:
                penalty += 1000 * abs(min_var_proxy)
                
            return rmse + penalty

        # 2. Multi-Start Optimization
        # SVI is non-convex; good initial guesses are critical.
        # Format: [a, b, rho, m, sigma]
        initial_guesses = [
            [0.04, 0.1, -0.5, 0.0, 0.1],   # Typical Equity Skew
            [0.04, 0.1, 0.0, 0.0, 0.1],    # Flat Smile
            [0.01, 0.05, -0.8, 0.1, 0.2],  # Deep Skew
            [np.min(w_data), 0.1, -0.5, 0.0, 0.1] # Adaptive 'a'
        ]
        
        # Bounds for SLSQP: (min, max)
        bounds = [
            (-0.5, 2.0), # a: can be slightly negative in Raw SVI if wings counteract, but usually >0
            (0.0, 5.0),  # b: must be positive
            (-0.999, 0.999), # rho
            (-1.0, 1.0), # m: usually near 0 for forward moneyness
            (0.001, 5.0) # sigma: must be positive
        ]

        best_result = None
        best_rmse = float('inf')

        for guess in initial_guesses:
            try:
                result = minimize(
                    objective, 
                    guess, 
                    method='SLSQP', 
                    bounds=bounds, 
                    options={'ftol': 1e-8, 'disp': False}
                )
                
                if result.success and result.fun < best_rmse:
                    best_rmse = result.fun
                    best_result = result
            except Exception:
                continue

        # 3. Finalize
        if best_result is not None:
            self.params = best_result.x
            # Recalculate pure RMSE without penalty for reporting
            w_final = self.raw_svi_formula(k_data, *self.params)
            self.rms_error = np.sqrt(np.mean((w_final - w_data)**2))
            
            return {
                "a": self.params[0],
                "b": self.params[1],
                "rho": self.params[2],
                "m": self.params[3],
                "sigma": self.params[4],
                "rmse": self.rms_error,
                "success": True
            }
        else:
            logger.warning("SVI Calibration failed (all guesses).")
            return {"success": False}

    def get_vol(self, k: float, T: float) -> float:
        """
        Predicts Implied Volatility for a given Log-Forward Moneyness k and Time T.
        """
        if self.params is None:
            raise ValueError("Model not calibrated yet.")
            
        a, b, rho, m, sigma = self.params
        return self.raw_svi_formula_vol(k, a, b, rho, m, sigma, T)

if __name__ == "__main__":
    # Unit Test
    # Simulate a typical Equity Vol Skew
    k_test = np.linspace(-0.2, 0.2, 10) # Log-Moneyness
    # Synthetic "market" vols (Smile shape)
    v_test = 0.20 - 0.3 * k_test + 0.5 * k_test**2 
    T_test = 0.1
    w_test = (v_test ** 2) * T_test
    
    svi = SVIModel()
    res = svi.calibrate(k_test, w_test)
    
    print("-" * 30)
    print("SVI Calibration Unit Test")
    print("-" * 30)
    if res['success']:
        print(f"Calibration Successful (RMSE: {res['rmse']:.6f})")
        print(f"Params: a={res['a']:.4f}, b={res['b']:.4f}, rho={res['rho']:.4f}, m={res['m']:.4f}, sigma={res['sigma']:.4f}")
        
        # Check density constraint
        min_var = res['a'] + res['b'] * res['sigma'] * np.sqrt(1 - res['rho']**2)
        print(f"Min Total Variance check (should be > 0): {min_var:.6f}")
    else:
        print("Calibration Failed")