 UPI Transaction Analyser

An interactive web app to analyse UPI transaction data — built with Python, Streamlit, and Machine Learning.

---

📊 Features

 Exploratory Data Analysis
- **Top Statistics** — Total transactions, total amount, success/failed count, average transaction value
- **Monthly Timeline** — Transaction volume trends over months
- **Daily Timeline** — Day-by-day spending patterns
- **Activity Map** — Busiest days of the week and months of the year
- **Hourly Pattern** — Which hours see the most transactions

 Transaction Insights
- **Amount Distribution** — Breakdown by ranges (<100, 100-500, 500-1K, 1K-5K, 5K-10K, >10K)
- **Category Analysis** — Spending split across Food, Shopping, Travel, Bills, etc.
- **UPI Bank Distribution** — @okaxis, @oksbi, @paytm, @gpay and more

 User Analysis
- **Top Senders & Receivers** — Who sends/receives the most money
- **WordCloud** — Most active users visualised

 Machine Learning
- **Anomaly Detection** — Suspicious/unusual transactions flagged using Isolation Forest
- **Success Prediction** — Random Forest model to predict transaction success with feature importance
- **Transaction Clustering** — KMeans clustering + PCA visualisation to find hidden patterns

---

 How to Use

1. Clone the repo and install dependencies
2. Generate sample data (or use your own CSV)
3. Run the app and upload your CSV
4. Select a user or keep **Overall**
5. Click **"Show Analysis"** 🎉

---

 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web app framework |
| Pandas | Data processing |
| Matplotlib | Charts and graphs |
| Scikit-learn | ML models (Random Forest, KMeans, IsolationForest, PCA) |
| NumPy | Numerical operations |
| WordCloud | Word cloud generation |

---

 Project Structure

```
upi_analyser/
│
├── app.py              # Main Streamlit app
├── helper.py           # All analysis & ML functions
├── preprocessor.py     # Data cleaning & feature extraction
├── generate_data.py    # Synthetic dataset generator
├── requirements.txt    # Dependencies
└── README.md
```

---

 CSV Format Required

Your uploaded CSV must have these columns:

| Column | Example |
|--------|---------|
| transaction_id | TXN000001 |
| timestamp | 2023-01-01 10:30:00 |
| sender_name | Rahul |
| sender_upi_id | rahul@okaxis |
| receiver_name | Priya |
| receiver_upi_id | priya@oksbi |
| amount | 500.00 |
| category | Food |
| status | SUCCESS / FAILED |

> Don't have real data? The app provides a **sample dataset download** on the home screen!

---

 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/ankito24/upi-transaction-analyser.git
cd upi-transaction-analyser

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample data (optional)
python generate_data.py

# 4. Run the app
streamlit run app.py
```

---

 ML Models Used

 Anomaly Detection — Isolation Forest
Flags transactions that are statistically unusual based on amount, hour, and month. Useful for detecting potential fraud or outliers.

 Success Prediction — Random Forest Classifier
Predicts whether a transaction will succeed or fail based on features like amount, hour, bank, and category. Also shows which features matter most.

 Clustering — KMeans + PCA
Groups similar transactions into clusters and visualises them in 2D using PCA dimensionality reduction.

---

 Developer

Made with ❤️ by **Ankit** — IIT Kanpur

[![GitHub](https://img.shields.io/badge/GitHub-ankito24-black?logo=github)](https://github.com/ankito24)
