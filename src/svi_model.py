import numpy as np
from scipy.optimize import minimize
import logging
from typing import List, Tuple, Dict

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
        # Ensure sigma is positive to prevent math errors
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    def calibrate(self, k_data: np.array, w_data: np.array) -> Dict:
        """
        Calibrates the SVI model to market data for a single slice (single expiry).
        
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
            
            # Penalize negative variance (impossible in reality)
            penalty = 0
            if np.any(w_model < 0):
                penalty = 1e6
                
            return np.sum((w_model - w_data)**2) + penalty

        # Initial constraints and bounds (Critical for stable calibration)
        # a: vertical shift
        # b: angle of asymptotes (b > 0)
        # rho: skew rotation (-1 < rho < 1)
        # m: horizontal shift
        # sigma: curvature (sigma > 0)
        
        initial_guess = [0.04, 0.1, -0.5, 0.0, 0.1]
        
        # Bounds: ((min, max), ...)
        bounds = (
            (None, None),   # a: Unbounded (though usually > 0)
            (0.0, None),    # b: Must be positive
            (-0.99, 0.99),  # rho: Correlation, strictly between -1 and 1
            (None, None),   # m: Unbounded
            (0.01, None)    # sigma: Must be positive
        )

        try:
            result = minimize(
                objective, 
                initial_guess, 
                method='L-BFGS-B', 
                bounds=bounds
            )
            
            self.params = result.x
            self.rms_error = np.sqrt(result.fun / len(k_data))
            
            logger.debug(f"Calibration successful. RMSE: {self.rms_error:.5f}")
            
            return {
                "a": self.params[0],
                "b": self.params[1],
                "rho": self.params[2],
                "m": self.params[3],
                "sigma": self.params[4],
                "rmse": self.rms_error,
                "success": result.success
            }
            
        except Exception as e:
            logger.error(f"SVI Calibration failed: {e}")
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
        
        if w < 0: 
            return 0.001 # Floor at small positive number
            
        return np.sqrt(w / T)

# --- Test Block ---
if __name__ == "__main__":
    # Fake data test
    k_test = np.array([-0.1, -0.05, 0.0, 0.05, 0.1]) # Log-moneyness
    v_test = np.array([0.22, 0.20, 0.18, 0.19, 0.21]) # Implied Vols
    T_test = 0.5
    w_test = v_test**2 * T_test # Convert to Total Variance
    
    svi = SVIModel()
    params = svi.calibrate(k_test, w_test)
    
    print("Calibrated Params:", params)
    print("Predicted Vol at ATM (k=0):", svi.get_vol(0, T_test))