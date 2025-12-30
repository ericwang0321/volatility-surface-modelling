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
    Responsible for raw data extraction and applying quantitative filtering rules.
    """

    def __init__(self, ticker_symbol: str = "SPY"):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        self.spot_price = None

    def fetch_spot_price(self) -> float:
        """
        Fetches the current live spot price of the underlying asset.
        """
        try:
            # 1. Try fetching from fast info
            price = self.ticker.info.get('regularMarketPrice')
            
            # 2. Fallback to historical data if 'info' is missing (common yfinance issue)
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
        Fetches option chains for all available expirations and returns a cleaned DataFrame.
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
                logger.warning(f"Skipping expiration {exp_date}: {e}")

        if not all_options:
            raise RuntimeError("No options data could be fetched.")

        # Concatenate all chains into one DataFrame
        raw_df = pd.concat(all_options, ignore_index=True)
        logger.info(f"Raw data fetched: {len(raw_df)} rows.")
        
        return self._clean_data(raw_df)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal cleaning logic to prepare data for SVI calibration.
        
        Steps:
        1. Parse dates with high precision.
        2. Filter for liquidity (Volume OR Open Interest).
        3. Select OTM (Out-of-the-Money) options only for better fit.
        """
        # 1. Standardize Expiration Date
        df['expirationDate'] = pd.to_datetime(df['expirationDate'])
        
        # 2. High-Precision Time to Maturity (T) calculation
        # Use UTC timestamp to prevent timezone issues
        now = pd.Timestamp.now()
        # Calculate total seconds difference and divide by seconds in a year (365 * 24 * 60 * 60)
        df['T'] = (df['expirationDate'] - now).dt.total_seconds() / 31536000.0
        
        # Filter out options that are expired or expiring today (< 0.5 days)
        df = df[df['T'] > 0.002].copy()

        # 3. Liquidity Filter (Relaxed Rule)
        # We keep the row if:
        # (Open Interest > 0 OR Volume > 0) AND (Bid > 0)
        # Bid > 0 ensures there is an active market maker quote.
        mask_liquidity = ((df['openInterest'] > 0) | (df['volume'] > 0)) & (df['bid'] > 0)
        df = df[mask_liquidity]

        # 4. Calculate Moneyness (Strike / Spot)
        df['S'] = self.spot_price
        df['K'] = df['strike']
        df['moneyness'] = df['K'] / df['S']

        # 5. OTM Selection (Smart Selection) - CRITICAL FOR SVI
        # For Calls: Keep only if Strike > Spot (Moneyness > 1.0) -> OTM Call
        # For Puts:  Keep only if Strike < Spot (Moneyness < 1.0) -> OTM Put
        # Why? ITM options often have wider spreads and less liquidity. 
        # The volatility of an ITM Call is theoretically the same as the OTM Put (Put-Call Parity),
        # so we just use the OTM Put data for that strike.
        condition_call = (df['optionType'] == 'call') & (df['moneyness'] >= 1.0)
        condition_put  = (df['optionType'] == 'put')  & (df['moneyness'] < 1.0)
        
        df = df[condition_call | condition_put].copy()

        # 6. Final Data Hygiene
        # Remove invalid implied volatility values (e.g., Yahoo sometimes returns 0 or negative junk)
        df = df[df['impliedVolatility'] > 0.001]
        
        # Select relevant columns
        cols_to_keep = [
            'contractSymbol', 'expirationDate', 'T', 'K', 'moneyness', 
            'optionType', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest'
        ]
        
        # Only keep columns that actually exist in the dataframe
        final_cols = [c for c in cols_to_keep if c in df.columns]
        
        logger.info(f"Cleaning complete. Final dataset: {len(df)} rows.")
        return df[final_cols]

if __name__ == "__main__":
    # Test Block
    loader = DataLoader("SPY")
    loader.fetch_spot_price()
    try:
        df = loader.fetch_options_chain()
        print(df.head())
        print(f"Call/Put Distribution:\n{df['optionType'].value_counts()}")
    except Exception as e:
        print(e)