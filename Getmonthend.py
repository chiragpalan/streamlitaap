
import pandas as pd

def get_next_month_end_date(given_date):
    # Ensure given_date is a Timestamp
    given_date = pd.to_datetime(given_date)

    # Extract year and month, then move to next month
    next_month = given_date.month + 1
    next_year = given_date.year

    # Handle December case (roll over to next year)
    if next_month > 12:
        next_month = 1
        next_year += 1

    # Get last date of next month
    next_month_end_date = pd.Timestamp(next_year, next_month, 1) + pd.offsets.MonthEnd(0)
    
    return next_month_end_date

# Sample DataFrame
df = pd.DataFrame({'date': ['30-07-2024', '15-06-2025', '31-12-2023']})

# Convert column to datetime format
df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")

# Apply function to the column
df['next_month_end_date'] = df['date'].apply(get_next_month_end_date)

# Convert back to desired string format if needed
df['next_month_end_date'] = df['next_month_end_date'].dt.strftime("%d-%m-%Y")

print(df)
