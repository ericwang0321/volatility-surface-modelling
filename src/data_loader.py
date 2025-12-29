import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles fetching and cleaning of options data from Yahoo Finance.
    """

    def __init__(self, ticker_symbol: str = "SPY"):
        """
        Args:
            ticker_symbol (str): The ticker to fetch (default: SPY).
        """
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        self.spot_price = None

    def fetch_spot_price(self) -> float:
        """
        Fetches the current live spot price of the underlying asset.
        """
        try:
            # Try fast retrieval
            price = self.ticker.info.get('regularMarketPrice')
            
            # Fallback to history if info is missing (common yfinance issue)
            if price is None:
                hist = self.ticker.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            if price is None:
                raise ValueError("Could not retrieve spot price.")
            
            self.spot_price = price
            logger.info(f"Current Spot Price for {self.ticker_symbol}: {self.spot_price:.2f}")
            return price
        except Exception as e:
            logger.error(f"Error fetching spot price: {e}")
            raise

    def fetch_options_chain(self) -> pd.DataFrame:
        """
        Fetches all available option chains (Calls & Puts) for all expiration dates.
        
        Returns:
            pd.DataFrame: A unified DataFrame containing clean options data.
        """
        if self.spot_price is None:
            self.fetch_spot_price()

        expirations = self.ticker.options
        logger.info(f"Found {len(expirations)} expiration dates. Starting download...")

        all_options = []

        for exp_date in expirations:
            try:
                # Fetch chain for specific expiration
                opt_chain = self.ticker.option_chain(exp_date)
                
                # Process Calls
                calls = opt_chain.calls.copy()
                calls['optionType'] = 'call'
                
                # Process Puts
                puts = opt_chain.puts.copy()
                puts['optionType'] = 'put'
                
                # Combine
                chain = pd.concat([calls, puts])
                chain['expirationDate'] = exp_date
                
                all_options.append(chain)
                
            except Exception as e:
                logger.warning(f"Failed to fetch data for expiration {exp_date}: {e}")

        if not all_options:
            raise RuntimeError("No options data could be fetched.")

        # Concatenate all chains into one DataFrame
        raw_df = pd.concat(all_options, ignore_index=True)
        logger.info(f"Raw data fetched: {len(raw_df)} rows.")
        
        return self._clean_data(raw_df)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to clean and filter the raw options DataFrame.
        """
        # 1. Standardize Expiration Date to Datetime64
        df['expirationDate'] = pd.to_datetime(df['expirationDate'])
        
        # 2. Calculate Days to Expiration
        # FIX: Use pd.Timestamp for 'today' to keep calculations vectorized in pandas
        today = pd.Timestamp.now().normalize() # normalize() sets time to 00:00:00
        
        # This results in a Timedelta Series, which supports .dt.days
        df['daysToExpiration'] = (df['expirationDate'] - today).dt.days
        
        # Filter out expired or today's options (T approx 0 causes math errors)
        df = df[df['daysToExpiration'] > 0].copy()
        
        # T = Days / 365.0 (Annualized time)
        df['T'] = df['daysToExpiration'] / 365.0

        # 3. Calculate Mid Price
        # Some rows might have 0 bid/ask, use 'lastPrice' as fallback if needed
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        
        # Filter: Remove rows where Mid Price is 0 or NaN
        df = df[df['mid_price'] > 0]

        # 4. Filter Liquid Contracts
        # Strict rule: Volume must be > 0. This removes stale quotes.
        df = df[df['volume'] > 0]

        # 5. Calculate Moneyness (Strike / Spot)
        # S = Spot, K = Strike
        df['S'] = self.spot_price
        df['K'] = df['strike']
        df['moneyness'] = df['K'] / df['S']

        # 6. Rename and Select columns
        cols_to_keep = [
            'contractSymbol', 'expirationDate', 'daysToExpiration', 'T', 
            'K', 'S', 'moneyness', 'optionType', 
            'bid', 'ask', 'mid_price', 
            'impliedVolatility', 'volume', 'openInterest'
        ]
        
        # Ensure all columns exist before selecting
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        clean_df = df[existing_cols].copy()
        
        logger.info(f"Data cleaning complete. Final dataset size: {len(clean_df)} rows.")
        return clean_df

# --- Test Block (Run this file directly to test) ---
if __name__ == "__main__":
    loader = DataLoader("SPY")
    spot = loader.fetch_spot_price()
    print(f"Spot: {spot}")
    
    try:
        df = loader.fetch_options_chain()
        print(df.head())
        print(df.describe())
    except Exception as e:
        print(f"Error during execution: {e}")