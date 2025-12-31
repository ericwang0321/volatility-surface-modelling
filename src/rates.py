import yfinance as yf
import pandas as pd
import numpy as np
from scipy import interpolate
import logging

# Configure logger
logger = logging.getLogger(__name__)

class RateProvider:
    """
    Provides the Risk-Free Rate Term Structure.
    Fetches US Treasury Yields (13W, 5Y, 10Y, 30Y) and builds an interpolation curve.
    """

    def __init__(self):
        # Tickers for US Treasury Yields
        self.tickers = {
            '13W': '^IRX',  # 13 Week Treasury Bill
            '5Y': '^FVX',   # 5 Year Treasury Note
            '10Y': '^TNX',  # 10 Year Treasury Note
            '30Y': '^TYX'   # 30 Year Treasury Bond
        }
        self.interpolator = None
        self._build_curve()

    def _build_curve(self):
        """
        Internal method to fetch data and construct the yield curve interpolator.
        """
        try:
            logger.info("Fetching Treasury Yield Curve data from Yahoo Finance...")
            
            time_map = {
                '13W': 13/52, 
                '5Y': 5.0,
                '10Y': 10.0,
                '30Y': 30.0
            }
            
            tenors = []
            rates = []
            
            # --- FIX: Use Sequential Fetching instead of bulk download ---
            # This avoids 'database is locked' errors common with yf.download threading
            for label, ticker_symbol in self.tickers.items():
                try:
                    # Fetch just the last day
                    ticker = yf.Ticker(ticker_symbol)
                    hist = ticker.history(period="5d")
                    
                    if not hist.empty:
                        # Get latest close
                        r_val = hist['Close'].iloc[-1]
                        
                        # Yahoo yields are in %, convert to decimal
                        if not pd.isna(r_val):
                            tenors.append(time_map[label])
                            rates.append(r_val / 100.0)
                    else:
                        logger.warning(f"No data found for {ticker_symbol}")
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {ticker_symbol}: {e}")

            if not rates:
                raise ValueError("No yield data could be fetched at all.")

            # Sort by tenor (time) to ensure correct interpolation
            # Combine into pairs, sort, then split back
            sorted_pairs = sorted(zip(tenors, rates))
            tenors = [t for t, r in sorted_pairs]
            rates = [r for t, r in sorted_pairs]

            # Convert to numpy arrays
            tenors = np.array(tenors)
            rates = np.array(rates)

            # Anchor at T=0 (Overnight rate ~= 13W rate)
            if len(tenors) > 0:
                tenors = np.insert(tenors, 0, 0.0)
                rates = np.insert(rates, 0, rates[0])

            # Create Linear Interpolator
            self.interpolator = interpolate.interp1d(
                tenors, rates, kind='linear', bounds_error=False, fill_value="extrapolate"
            )
            
            logger.info(f"Yield Curve constructed using {len(rates)-1} points. 10Y Rate: {rates[-2]:.2%}")

        except Exception as e:
            logger.error(f"Critical Error building yield curve: {e}. Using flat 4% fallback.")
            self.interpolator = lambda x: 0.04

    def get_risk_free_rate(self, T: float) -> float:
        """
        Returns the interpolated risk-free rate for a specific maturity T.
        """
        if self.interpolator is None:
            return 0.04
        T = max(0.0, T)
        return float(self.interpolator(T))

if __name__ == "__main__":
    # Test Block
    logging.basicConfig(level=logging.INFO)
    rp = RateProvider()
    
    print("-" * 30)
    print("Testing Rate Provider (Fix Applied):")
    # Test points spanning short to long term
    for t_test in [1/12, 0.5, 1.0, 5.0, 10.0, 20.0]:
        r = rp.get_risk_free_rate(t_test)
        print(f"Time: {t_test:.2f} years -> Rate: {r:.4%}")
    print("-" * 30)