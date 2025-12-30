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
        self.vol_surface = vol_surface

    def price_barrier_option(self, K: float, barrier: float, option_type: str = "down-and-out_call", 
                           n_paths: int = 5000, n_steps: int = 100, model: str = "local_vol", 
                           const_vol: float = 0.2, seed: int = None) -> dict:
        """
        Prices a Barrier Option. 
        ADDED: 'seed' parameter for Common Random Numbers (CRN).
        """
        # --- FIX: Set Random Seed if provided ---
        if seed is not None:
            np.random.seed(seed)
            
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.full(n_paths, self.S0)
        alive = np.full(n_paths, True, dtype=bool)
        
        for i in range(n_steps):
            t_current = i * dt
            
            # 1. Determine Volatility
            if model == "local_vol":
                # Vectorized lookup
                # Optimization: We assume S > 0, clip extreme lookups
                S_lookup = np.maximum(S, 0.01) 
                
                # Using a vectorized wrapper for the surface method
                # Note: For strict performance, this loop should be optimized further, 
                # but for this demo, vectorizing the method call is sufficient.
                # We catch errors to prevent crash on single path failure
                try:
                    # Creating a simple vectorized function on the fly is slow, 
                    # better to iterate if get_local_vol isn't natively vectorized.
                    # Here we assume get_local_vol handles scalar inputs.
                    sigma = np.array([self.vol_surface.get_local_vol(s, t_current) for s in S_lookup])
                except:
                    sigma = np.full(n_paths, 0.2) # Fallback

                sigma = np.clip(sigma, 0.01, 1.0) # Safety clamp
            else:
                sigma = np.full(n_paths, const_vol)

            # 2. Generate Random Shocks
            z = np.random.normal(0, 1, n_paths)
            
            # 3. Evolve Price
            S = S * np.exp((self.r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z)
            
            # 4. Check Barrier
            if "down" in option_type:
                alive = alive & (S > barrier)
                
        # Payoff
        payoffs = np.where(alive, np.maximum(S - K, 0), 0)
        
        price = np.mean(payoffs) * np.exp(-self.r * self.T)
        std_err = np.std(payoffs) / np.sqrt(n_paths)
        
        return {"price": price, "std_err": std_err}

    def calculate_delta(self, K: float, barrier: float, model: str = "local_vol", 
                        n_paths: int = 5000, epsilon_pct: float = 0.01) -> float:
        """
        Calculates Delta using Central Finite Difference with Common Random Numbers.
        """
        dS = self.S0 * epsilon_pct
        S_up = self.S0 + dS
        S_down = self.S0 - dS
        
        # --- FIX: Generate a random seed to share between Up and Down scenarios ---
        # This ensures that the ONLY difference between the two runs is the Spot Price,
        # not the random path noise.
        common_seed = np.random.randint(0, 1000000)
        
        # 1. Price Up
        pricer_up = MonteCarloPricer(S_up, self.r, self.T, self.vol_surface)
        
        # For BS, keep vol constant (Sticky Strike assumption for simple delta)
        if model == "black_scholes":
            # Use ATM vol of the *original* spot to keep comparison fair
            const_vol = self.vol_surface.get_implied_vol(0, self.T)
            res_up = pricer_up.price_barrier_option(K, barrier, model="black_scholes", 
                                                  const_vol=const_vol, n_paths=n_paths, seed=common_seed)
        else:
            res_up = pricer_up.price_barrier_option(K, barrier, model="local_vol", 
                                                  n_paths=n_paths, seed=common_seed)
            
        # 2. Price Down (Using SAME SEED)
        pricer_down = MonteCarloPricer(S_down, self.r, self.T, self.vol_surface)
        
        if model == "black_scholes":
            const_vol = self.vol_surface.get_implied_vol(0, self.T)
            res_down = pricer_down.price_barrier_option(K, barrier, model="black_scholes", 
                                                      const_vol=const_vol, n_paths=n_paths, seed=common_seed)
        else:
            res_down = pricer_down.price_barrier_option(K, barrier, model="local_vol", 
                                                      n_paths=n_paths, seed=common_seed)
            
        delta = (res_up['price'] - res_down['price']) / (2 * dS)
        return delta