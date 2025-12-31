import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

class MonteCarloPricer:
    """
    Pricing Engine using Monte Carlo Simulation.
    Supports Local Volatility with Dynamic Rates (Term Structure) and Dividends.
    """

    def __init__(self, S0: float, T: float, rate_provider, q: float = 0.0, vol_surface=None):
        """
        Args:
            S0: Current Spot Price
            T: Time to Maturity (years)
            rate_provider: Instance of RateProvider (for dynamic r)
            q: Dividend Yield (decimal, e.g., 0.015)
            vol_surface: Instance of VolatilitySurface
        """
        self.S0 = S0
        self.T = T
        self.rate_provider = rate_provider
        self.q = q
        self.vol_surface = vol_surface

    def price_barrier_option(self, K: float, barrier: float, option_type: str = "down-and-out_call", 
                           n_paths: int = 5000, n_steps: int = 100, model: str = "local_vol", 
                           const_vol: float = 0.2, seed: int = None) -> dict:
        """
        Prices a Barrier Option.
        Uses r(T) from the yield curve and continuous dividend yield q.
        """
        if seed is not None:
            np.random.seed(seed)
            
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # 1. Get Dynamic Risk-Free Rate for this specific maturity T
        # This is the "r" corresponding to the option's expiry
        r_dynamic = self.rate_provider.get_risk_free_rate(self.T)
        
        # 2. Calculate Drift Base (mu)
        # In risk-neutral pricing with dividends: drift = r - q
        drift_base = r_dynamic - self.q
        
        S = np.full(n_paths, self.S0)
        alive = np.full(n_paths, True, dtype=bool)

        # Check if Vol Surface is ready for Local Vol
        if model == "local_vol":
            if self.vol_surface is None:
                raise ValueError("Vol Surface required for Local Vol pricing.")
            if self.vol_surface.interpolator is None:
                self.vol_surface.build_local_vol_interpolator()
        
        # --- Monte Carlo Loop ---
        for i in range(n_steps):
            t_current = i * dt
            
            # A. Determine Volatility (sigma)
            if model == "local_vol":
                # Vectorized lookup from the pre-computed grid
                # Clip S to stay within grid bounds to prevent crashes
                S_clipped = np.clip(S, self.vol_surface.interp_s_grid[0], self.vol_surface.interp_s_grid[-1])
                t_vec = np.full(n_paths, t_current)
                points = np.column_stack((S_clipped, t_vec))
                
                sigma = self.vol_surface.get_local_vol_fast(points)
                sigma = np.clip(sigma, 0.01, 2.0) # Safety clamp
            else:
                sigma = np.full(n_paths, const_vol)

            # B. Generate Random Motion
            z = np.random.normal(0, 1, n_paths)
            
            # C. Evolve Price
            # Geometric Brownian Motion with Dividends:
            # dS = (r - q)S dt + sigma S dW
            # Discretized: S_new = S * exp( (r - q - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z )
            S = S * np.exp((drift_base - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z)
            
            # D. Check Barrier
            if "down" in option_type:
                alive = alive & (S > barrier)
            elif "up" in option_type:
                alive = alive & (S < barrier)
                
        # Payoff
        # Call: max(S-K, 0), Put: max(K-S, 0)
        if "call" in option_type.lower():
            payoffs = np.where(alive, np.maximum(S - K, 0), 0)
        else:
            payoffs = np.where(alive, np.maximum(K - S, 0), 0)
        
        # Discount back using the risk-free rate r_dynamic
        price = np.mean(payoffs) * np.exp(-r_dynamic * self.T)
        std_err = np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            "price": price, 
            "std_err": std_err, 
            "used_r": r_dynamic,
            "used_q": self.q
        }

    def calculate_delta(self, K: float, barrier: float, model: str = "local_vol", 
                        n_paths: int = 5000, epsilon_pct: float = 0.01) -> float:
        """
        Calculates Delta.
        Ensures new pricer instances inherit the correct RateProvider and q.
        """
        dS = self.S0 * epsilon_pct
        S_up = self.S0 + dS
        S_down = self.S0 - dS
        
        # Use Common Random Numbers (CRN) for variance reduction
        common_seed = np.random.randint(0, 1000000)
        
        # Create Pricers for bumped spots
        # IMPORTANT: Pass self.rate_provider and self.q to preserve environment
        pricer_up = MonteCarloPricer(S_up, self.T, self.rate_provider, self.q, self.vol_surface)
        pricer_down = MonteCarloPricer(S_down, self.T, self.rate_provider, self.q, self.vol_surface)
        
        # Setup Volatility params
        if model == "black_scholes":
            const_vol = self.vol_surface.get_implied_vol(0, self.T) if self.vol_surface else 0.2
            res_up = pricer_up.price_barrier_option(K, barrier, model="black_scholes", const_vol=const_vol, n_paths=n_paths, seed=common_seed)
            res_down = pricer_down.price_barrier_option(K, barrier, model="black_scholes", const_vol=const_vol, n_paths=n_paths, seed=common_seed)
        else:
            res_up = pricer_up.price_barrier_option(K, barrier, model="local_vol", n_paths=n_paths, seed=common_seed)
            res_down = pricer_down.price_barrier_option(K, barrier, model="local_vol", n_paths=n_paths, seed=common_seed)
            
        delta = (res_up['price'] - res_down['price']) / (2 * dS)
        return delta