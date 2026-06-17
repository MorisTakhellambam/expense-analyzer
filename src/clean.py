import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DATA_PATH, CLEANED_DATA_PATH

def cleanData() -> pd.DataFrame:
    """
    Imports required columns and cleans the raw data by removing negative signs and commas from the 'amount' column.
    
    Returns:
        pd.DataFrame: A cleaned DataFrame with the specified columns.
    """

    # Read the raw data
    df = pd.read_csv(RAW_DATA_PATH, usecols=['date', 'account', 'category', 'amount', 'description'])
    
    # Remove negative sign and commas from amount column
    df['amount'] = df['amount'].astype(str).str.replace('-', '', regex=False)
    df['amount'] = df['amount'].astype(str).str.replace(',', '', regex=False).astype(float)
    
    return df

def saveCleanedData(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the cleaned DataFrame to a specified CSV file.
    
    Args:
        df (pd.DataFrame): The cleaned DataFrame to be saved.
        output_path (str): The path where the cleaned data will be saved.
    """

    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Clean the data
    cleaned_df = cleanData()
    
    # Save the cleaned data
    saveCleanedData(cleaned_df, CLEANED_DATA_PATH)
