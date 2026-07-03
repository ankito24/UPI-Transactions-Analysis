import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

first_names = ["Rahul","Priya","Amit","Sneha","Rohan","Ananya","Vikas","Pooja",
                "Karan","Divya","Arjun","Nisha","Suresh","Kavya","Raj","Meera",
                "Aditya","Simran","Nikhil","Riya","Varun","Tanya","Siddharth","Ishaan"]

banks = ["@okaxis","@oksbi","@okhdfcbank","@okicici","@ybl","@upi","@paytm","@gpay"]
categories = ["Food","Shopping","Recharge","Bills","Transfer","Entertainment","Travel","Education"]

def generate_upi_id(name):
    return name.lower().replace(" ","") + str(random.randint(1,999)) + random.choice(banks)

n = 5000
senders   = random.choices(first_names, k=n)
receivers = random.choices(first_names, k=n)

start = datetime(2023, 1, 1)
timestamps = [start + timedelta(minutes=random.randint(0, 525600)) for _ in range(n)]

amounts = np.concatenate([
    np.random.exponential(scale=500, size=4500),
    np.random.uniform(10000, 50000, size=500)
])
np.random.shuffle(amounts)
amounts = np.clip(amounts, 1, 100000).round(2)

status = np.random.choice(["SUCCESS","FAILED"], size=n, p=[0.85, 0.15])

df = pd.DataFrame({
    "transaction_id" : [f"TXN{str(i).zfill(6)}" for i in range(n)],
    "timestamp"      : timestamps,
    "sender_name"    : senders,
    "sender_upi_id"  : [generate_upi_id(s) for s in senders],
    "receiver_name"  : receivers,
    "receiver_upi_id": [generate_upi_id(r) for r in receivers],
    "amount"         : amounts,
    "category"       : random.choices(categories, k=n),
    "status"         : status,
})

df = df.sort_values("timestamp").reset_index(drop=True)
df.to_csv("upi_transactions.csv", index=False)
print(f"Dataset generated: {len(df)} transactions")
print(df.head())
