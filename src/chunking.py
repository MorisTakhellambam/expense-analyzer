import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CLEANED_DATA_PATH, CLEANED_DATA_PATH_WITH_TEXT

def row_to_text(row: pd.Series) -> str:
    """
    Converts a single row of the cleaned data into a text format suitable for language model processing.
    
    Args:
        row (pd.Series): A row from the cleaned DataFrame.
    
    Returns:
        str: The text representation of the row.
    """
    
    date = row['date']              # 'date' is already a string in the format 'DD-MM-YYYY' after cleaning

    row['date'] = pd.to_datetime(row['date'], errors='coerce')      # to be able to use strftime to get the day of the week
    day = row['date'].strftime('%A')

    return(
        f"On {day}, {date}, you spent {row['amount']:.2f} {row['currency']} on {row['category']}. The description of the transaction is: '{row['description']}'."
    )

def add_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'text' column to the DataFrame, where each row is converted to a text format.
    
    Args:
        df (pd.DataFrame): The cleaned DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with the added 'text' column.
    """
    df['text'] = df.apply(row_to_text, axis=1)

    return df

if __name__ == "__main__":
    # Load the cleaned data
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # Add the text column
    df_with_text = add_text_column(df)
    
    # Save the updated DataFrame back to CSV
    df_with_text.to_csv(CLEANED_DATA_PATH_WITH_TEXT, index=False)
    print(f"Updated data with text column saved to {CLEANED_DATA_PATH_WITH_TEXT}")