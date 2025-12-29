import numpy as np
from scipy.stats import norm
import logging

# Configure logger
logger = logging.getLogger(__name__)

class MonteCarloPricer:
    """
    Pricing Engine using Monte Carlo Simulation.
    Supports both Black-Scholes (Constant Vol) and Local Volatility models.
    """

    def __init__(self, S0: float, r: float, T: float, vol_surface=None):
        self.S0 = S0
        self.r = r
        self.T = T
        self.vol_surface = vol_surface  # Instance of VolatilitySurface

    def price_bs_call(self, K: float, sigma: float) -> float:
        """Closed-form Black-Scholes price for European Call."""
        d1 = (np.log(self.S0 / K) + (self.r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        return self.S0 * norm.cdf(d1) - K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def price_barrier_option(self, K: float, barrier: float, option_type: str = "down-and-out_call", 
                           n_paths: int = 5000, n_steps: int = 100, model: str = "local_vol", const_vol: float = 0.2) -> dict:
        """
        Prices a Barrier Option using Monte Carlo.
        
        Args:
            K: Strike Price
            barrier: Barrier Level (e.g., if price touches this, option dies)
            model: 'local_vol' or 'black_scholes'
            const_vol: Volatility used if model is 'black_scholes'
            
        Returns:
            Dictionary with 'price' and 'std_error'
        """
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize paths starting at S0
        # Shape: [n_paths]
        S = np.full(n_paths, self.S0)
        
        # Track if barrier is hit (True = Alive, False = Knocked Out)
        alive = np.full(n_paths, True, dtype=bool)
        
        for i in range(n_steps):
            t_current = i * dt
            
            # 1. Determine Volatility for this step
            if model == "local_vol":
                # We need to look up sigma_loc(S_t, t) for EACH path
                # Since get_local_vol is scalar, we use a vectorized mapping or simple loop for clarity here.
                # Optimization: In production, we'd vectorize the surface lookup. 
                # Here we approximate by using the mean S of active paths to speed up demo, 
                # OR (Better) strictly loop. Let's strictly loop but optimize with a helper if possible.
                
                # To keep it fast for Python: Pre-calculate grid or use vector function?
                # Let's use a simple list comprehension for the active paths.
                
                # Note: This can be slow in pure Python. For 5000 paths it's okay.
                # We use a vectorized wrapper around our surface method.
                v_get_vol = np.vectorize(lambda s: self.vol_surface.get_local_vol(s, t_current))
                sigma = v_get_vol(S)
                
                # Clamp extreme vols for stability
                sigma = np.clip(sigma, 0.05, 1.0)
                
            else:
                sigma = np.full(n_paths, const_vol)

            # 2. Generate Random Shocks
            z = np.random.normal(0, 1, n_paths)
            
            # 3. Evolve Price (Geometric Brownian Motion)
            # S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            S = S * np.exp((self.r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z)
            
            # 4. Check Barrier Condition (Down-and-Out)
            if "down" in option_type:
                # If Price <= Barrier, it's knocked out
                alive = alive & (S > barrier)
                
        # Payoff at maturity
        # If alive: max(S - K, 0), else: 0
        payoffs = np.where(alive, np.maximum(S - K, 0), 0)
        
        # Discount back to present
        price = np.mean(payoffs) * np.exp(-self.r * self.T)
        std_err = np.std(payoffs) / np.sqrt(n_paths)
        
        return {"price": price, "std_err": std_err}

    def calculate_delta(self, K: float, barrier: float, model: str = "local_vol", 
                        n_paths: int = 5000, epsilon_pct: float = 0.01) -> float:
        """
        Calculates Delta using Central Finite Difference with Common Random Numbers.
        Delta = (Price(S_up) - Price(S_down)) / (2 * dS)
        """
        # Shock spot price up and down
        dS = self.S0 * epsilon_pct
        S_up = self.S0 + dS
        S_down = self.S0 - dS
        
        # We need to temporarily construct pricers with shocked spots
        # IMPORTANT: Use the SAME random seed inside price_barrier_option to reduce variance
        # (Current implementation allows numpy random state control? 
        #  For simplicity here, we assume large n_paths averages out, 
        #  but ideally we fix the seed explicitly in the pricer loop).
        
        # Hack: Since our pricer generates new random numbers each time, 
        # we need a sufficiently large n_paths to stabilize Delta.
        
        # 1. Price Up
        pricer_up = MonteCarloPricer(S_up, self.r, self.T, self.vol_surface)
        # Note: For BS, we assume Vol stays constant. For Local Vol, surface stays same, but we lookup at new S.
        if model == "black_scholes":
            # Recalculate ATM vol for the new spot? Or keep fixed? 
            # Standard Greek calculation keeps implied vol fixed (Sticky Strike vs Sticky Delta).
            # Let's assume constant vol (Sticky Strike).
            const_vol = self.vol_surface.get_implied_vol(np.log(self.S0/self.S0), self.T) # ATM of base
            res_up = pricer_up.price_barrier_option(K, barrier, model="black_scholes", const_vol=const_vol, n_paths=n_paths)
        else:
            res_up = pricer_up.price_barrier_option(K, barrier, model="local_vol", n_paths=n_paths)
            
        # 2. Price Down
        pricer_down = MonteCarloPricer(S_down, self.r, self.T, self.vol_surface)
        if model == "black_scholes":
            const_vol = self.vol_surface.get_implied_vol(np.log(self.S0/self.S0), self.T)
            res_down = pricer_down.price_barrier_option(K, barrier, model="black_scholes", const_vol=const_vol, n_paths=n_paths)
        else:
            res_down = pricer_down.price_barrier_option(K, barrier, model="local_vol", n_paths=n_paths)
            
        delta = (res_up['price'] - res_down['price']) / (2 * dS)
        return delta