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
    """

    def __init__(self):
        # Parameters: [a, b, rho, m, sigma]
        self.params = None
        self.rms_error = None

    @staticmethod
    def raw_svi_formula(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
        """
        The Raw SVI formula: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        
        Args:
            k: Log-moneyness log(Strike/Spot)
            a, b, rho, m, sigma: SVI parameters
            
        Returns:
            Total Variance (w) = sigma_implied^2 * T
        """
        # SVI Formula
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    def calibrate(self, k_data: np.array, w_data: np.array) -> Optional[Dict]:
        """
        Calibrates the SVI model to market data for a single slice (single expiry).
        Uses Multi-Start Optimization to handle local minima in short-dated options.
        
        Args:
            k_data: Array of log-moneyness log(K/S)
            w_data: Array of total variance (implied_vol^2 * T)
            
        Returns:
            Dict containing calibrated parameters and optimization status.
        """
        # Objective function: Minimize Sum of Squared Errors (SSE)
        def objective(params):
            a, b, rho, m, sigma = params
            w_model = self.raw_svi_formula(k_data, a, b, rho, m, sigma)
            
            # Penalty for NaN or negative variance (impossible in reality)
            if np.any(np.isnan(w_model)) or np.any(w_model < 0):
                return 1e9
                
            return np.sum((w_model - w_data)**2)

        # Bounds: a, b, rho, m, sigma
        # Relaxed 'a' slightly, strict 'rho' and 'sigma'
        bounds = [
            (-0.5, 2.0),      # a: level (usually >0, but can be slightly neg mathematically if w>0)
            (0.0, 5.0),       # b: angle (must be positive)
            (-0.999, 0.999),  # rho: Correlation (strictly between -1 and 1)
            (-2.0, 2.0),      # m: horizontal shift
            (0.001, 2.0)      # sigma: curvature (must be positive)
        ]

        # --- Multi-Start Strategy (The Fix) ---
        # We try multiple initial guesses. If one fails, we try the next.
        # This is crucial for short-dated options which have extreme Skew.
        initial_guesses = [
            [0.04, 0.1, -0.5, 0.0, 0.1],   # 1. Standard (Gentle Smile)
            [0.04, 0.3, -0.9, -0.1, 0.05], # 2. High Skew (Short-term Put heavy)
            [0.01, 0.1, 0.0, 0.0, 0.2],    # 3. High Curvature (Earnings/Event)
            [0.10, 0.05, -0.3, 0.1, 0.1]   # 4. High Vol Level
        ]

        best_result = None
        best_mse = float('inf')

        for i, guess in enumerate(initial_guesses):
            try:
                # SLSQP often handles boundary constraints better than L-BFGS-B for SVI
                result = minimize(
                    objective, 
                    guess, 
                    method='SLSQP', 
                    bounds=bounds,
                    tol=1e-8,
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    mse = result.fun
                    if mse < best_mse:
                        best_mse = mse
                        best_result = result
            except Exception:
                continue
        
        # Check if we found ANY valid solution
        if best_result is not None:
            self.params = best_result.x
            self.rms_error = np.sqrt(best_mse / len(k_data))
            
            # Determine success based on RMSE thresholds (optional, but good for logs)
            # If RMSE is too huge, it's technically a math success but a model failure.
            is_acceptable = self.rms_error < 0.1 

            return {
                "a": self.params[0],
                "b": self.params[1],
                "rho": self.params[2],
                "m": self.params[3],
                "sigma": self.params[4],
                "rmse": self.rms_error,
                "success": True # We mark it true if optimizer converged
            }
        else:
            logger.warning(f"SVI Calibration failed for slice (all guesses failed).")
            return None

    def get_vol(self, k: float, T: float) -> float:
        """
        Predicts Implied Volatility for a given Log-Moneyness and Time.
        sigma_impl = sqrt(w / T)
        """
        if self.params is None:
            raise ValueError("Model not calibrated yet.")
            
        a, b, rho, m, sigma = self.params
        w = self.raw_svi_formula(k, a, b, rho, m, sigma)
        
        # Safety floor: Variance cannot be negative.
        # If model implies negative variance, floor at small epsilon.
        if w < 1e-6: 
            w = 1e-6
            
        return np.sqrt(w / T)

# --- Test Block ---
if __name__ == "__main__":
    # Test with typical short-dated high-skew data
    k_test = np.array([-0.2, -0.1, 0.0, 0.1, 0.2]) 
    v_test = np.array([0.40, 0.30, 0.20, 0.18, 0.17]) # High skew
    T_test = 0.02 # 1 week
    w_test = v_test**2 * T_test 
    
    svi = SVIModel()
    params = svi.calibrate(k_test, w_test)
    
    if params:
        print("Calibrated Params:", params)
        print("RMSE:", params['rmse'])
        print("Predicted Vol at ATM (k=0):", svi.get_vol(0, T_test))
    else:
        print("Calibration Failed")