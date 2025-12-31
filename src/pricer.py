import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

class MonteCarloPricer:
    """
    Pricing Engine using Monte Carlo Simulation.
    Supports both Black-Scholes (Constant Vol) and Local Volatility models.
    Updated to use Fast Grid Interpolation for Local Volatility.
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
        Uses Grid Interpolation for Local Volatility to improve performance.
        """
        if seed is not None:
            np.random.seed(seed)
            
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.full(n_paths, self.S0)
        alive = np.full(n_paths, True, dtype=bool)

        # --- OPTIMIZATION: Ensure Interpolator is built ---
        if model == "local_vol":
            if self.vol_surface is None:
                raise ValueError("Vol Surface required for Local Vol pricing.")
            if self.vol_surface.interpolator is None:
                self.vol_surface.build_local_vol_interpolator()
        
        for i in range(n_steps):
            t_current = i * dt
            
            # 1. Determine Volatility
            if model == "local_vol":
                # --- Vectorized Fast Lookup ---
                # Create coordinate pairs [S, t] for all paths at once
                # Shape: (n_paths, 2)
                
                # Safety: Clip S to ensure it stays within interpolation bounds
                # (Optional, but good practice. Using reasonable grid limits usually sufficient)
                S_clipped = np.clip(S, self.vol_surface.interp_s_grid[0], self.vol_surface.interp_s_grid[-1])
                
                # Broadcast time t_current to match dimension of S
                t_vec = np.full(n_paths, t_current)
                
                # Stack into (N, 2) array
                points = np.column_stack((S_clipped, t_vec))
                
                # Query interpolator (O(1) operation per point, vectorized)
                sigma = self.vol_surface.get_local_vol_fast(points)
                
                # Safety clamp for sigma
                sigma = np.clip(sigma, 0.01, 1.0) 
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
        
        # --- Common Random Numbers (CRN) ---
        common_seed = np.random.randint(0, 1000000)
        
        # 1. Price Up
        pricer_up = MonteCarloPricer(S_up, self.r, self.T, self.vol_surface)
        
        if model == "black_scholes":
            if self.vol_surface:
                const_vol = self.vol_surface.get_implied_vol(0, self.T)
            else:
                const_vol = 0.2
            res_up = pricer_up.price_barrier_option(K, barrier, model="black_scholes", 
                                                  const_vol=const_vol, n_paths=n_paths, seed=common_seed)
        else:
            res_up = pricer_up.price_barrier_option(K, barrier, model="local_vol", 
                                                  n_paths=n_paths, seed=common_seed)
            
        # 2. Price Down (Using SAME SEED)
        pricer_down = MonteCarloPricer(S_down, self.r, self.T, self.vol_surface)
        
        if model == "black_scholes":
            # Re-use same const_vol for fair comparison
            if self.vol_surface:
                const_vol = self.vol_surface.get_implied_vol(0, self.T)
            else:
                const_vol = 0.2
            res_down = pricer_down.price_barrier_option(K, barrier, model="black_scholes", 
                                                      const_vol=const_vol, n_paths=n_paths, seed=common_seed)
        else:
            res_down = pricer_down.price_barrier_option(K, barrier, model="local_vol", 
                                                      n_paths=n_paths, seed=common_seed)
            
        delta = (res_up['price'] - res_down['price']) / (2 * dS)
        return delta