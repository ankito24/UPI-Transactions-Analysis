import pandas as pd

def preprocess(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['year']      = df['timestamp'].dt.year
    df['month_num'] = df['timestamp'].dt.month
    df['month']     = df['timestamp'].dt.month_name()
    df['day']       = df['timestamp'].dt.day
    df['day_name']  = df['timestamp'].dt.day_name()
    df['hour']      = df['timestamp'].dt.hour
    df['date_only'] = df['timestamp'].dt.date

    df['sender_bank']   = df['sender_upi_id'].str.extract(r'(@\w+)$')
    df['receiver_bank'] = df['receiver_upi_id'].str.extract(r'(@\w+)$')

    df['amount'] = df['amount'].astype(float)

    return df
