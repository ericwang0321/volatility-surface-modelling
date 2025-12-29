import yfinance as yf
import logging

# Configure logger
logger = logging.getLogger(__name__)

class RateProvider:
    """
    Provides risk-free interest rates for option pricing.
    Current implementation fetches the 13-Week Treasury Bill (^IRX) yield.
    """

    def __init__(self):
        self.ticker_symbol = "^IRX"  # 13 Week Treasury Bill

    def get_risk_free_rate(self) -> float:
        """
        Fetches the latest risk-free rate from Yahoo Finance.
        
        Returns:
            float: The risk-free rate in decimal format (e.g., 0.045 for 4.5%).
                   Returns 0.04 (4%) as fallback if fetch fails.
        """
        try:
            logger.info(f"Fetching risk-free rate from {self.ticker_symbol}...")
            ticker = yf.Ticker(self.ticker_symbol)
            
            # Get the most recent closing price
            todays_data = ticker.history(period="1d")
            
            if not todays_data.empty:
                # The yield is quoted in percentage (e.g., 4.25), so divide by 100
                rate = todays_data['Close'].iloc[-1] / 100.0
                logger.info(f"Risk-free rate fetched: {rate:.4f}")
                return rate
            else:
                logger.warning("Empty data for rates. Using fallback 4.0%.")
                return 0.04

        except Exception as e:
            logger.error(f"Failed to fetch risk-free rate: {e}")
            return 0.04  # Fallback to 4%