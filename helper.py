import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from wordcloud import WordCloud

# ── filter helper ──────────────────────────────────────────────────────────────
def filter_user(df, selected_user):
    if selected_user != "Overall":
        mask = (df['sender_name'] == selected_user) | (df['receiver_name'] == selected_user)
        df = df[mask]
    return df

# ── top statistics ─────────────────────────────────────────────────────────────
def fetch_stats(selected_user, df):
    df = filter_user(df, selected_user)
    total_tx      = df.shape[0]
    total_amount  = df['amount'].sum()
    success_count = df[df['status'] == 'SUCCESS'].shape[0]
    failed_count  = df[df['status'] == 'FAILED'].shape[0]
    avg_amount    = df['amount'].mean()
    return total_tx, round(total_amount, 2), success_count, failed_count, round(avg_amount, 2)

# ── monthly timeline ───────────────────────────────────────────────────────────
def monthly_timeline(selected_user, df):
    df = filter_user(df, selected_user)
    timeline = df.groupby(['year','month_num','month'])['amount'].sum().reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline

# ── daily timeline ─────────────────────────────────────────────────────────────
def daily_timeline(selected_user, df):
    df = filter_user(df, selected_user)
    daily = df.groupby('date_only')['amount'].sum().reset_index()
    return daily

# ── activity maps ──────────────────────────────────────────────────────────────
def week_activity_map(selected_user, df):
    df = filter_user(df, selected_user)
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    counts = df['day_name'].value_counts().reindex(order).fillna(0)
    return counts

def month_activity_map(selected_user, df):
    df = filter_user(df, selected_user)
    return df['month'].value_counts()

def hourly_activity(selected_user, df):
    df = filter_user(df, selected_user)
    return df.groupby('hour')['amount'].sum()

# ── success vs failed ──────────────────────────────────────────────────────────
def status_breakdown(selected_user, df):
    df = filter_user(df, selected_user)
    return df['status'].value_counts()

# ── top senders & receivers ────────────────────────────────────────────────────
def top_senders(df, n=10):
    return df.groupby('sender_name')['amount'].sum().sort_values(ascending=False).head(n)

def top_receivers(df, n=10):
    return df.groupby('receiver_name')['amount'].sum().sort_values(ascending=False).head(n)

# ── bank distribution ──────────────────────────────────────────────────────────
def bank_distribution(selected_user, df):
    df = filter_user(df, selected_user)
    return df['sender_bank'].value_counts()

# ── category analysis ──────────────────────────────────────────────────────────
def category_analysis(selected_user, df):
    df = filter_user(df, selected_user)
    return df.groupby('category')['amount'].sum().sort_values(ascending=False)

# ── amount distribution bins ───────────────────────────────────────────────────
def amount_bins(selected_user, df):
    df = filter_user(df, selected_user)
    bins   = [0, 100, 500, 1000, 5000, 10000, 100000]
    labels = ['<100','100-500','500-1K','1K-5K','5K-10K','>10K']
    df['bin'] = pd.cut(df['amount'], bins=bins, labels=labels)
    return df['bin'].value_counts().reindex(labels)

# ── anomaly detection ──────────────────────────────────────────────────────────
def detect_anomalies(selected_user, df):
    df = filter_user(df, selected_user).copy()
    X = df[['amount','hour','month_num']].fillna(0)
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = iso.fit_predict(X)
    anomalies = df[df['anomaly'] == -1][['timestamp','sender_name','receiver_name','amount','status']].head(20)
    return anomalies

# ── random forest prediction ───────────────────────────────────────────────────
def predict_success(df):
    df = df.copy()
    le = LabelEncoder()
    df['sender_bank_enc']   = le.fit_transform(df['sender_bank'].fillna('unknown'))
    df['receiver_bank_enc'] = le.fit_transform(df['receiver_bank'].fillna('unknown'))
    df['category_enc']      = le.fit_transform(df['category'].fillna('unknown'))
    df['status_enc']        = (df['status'] == 'SUCCESS').astype(int)

    features = ['amount','hour','month_num','sender_bank_enc','receiver_bank_enc','category_enc']
    X = df[features]
    y = df['status_enc']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importance_df = pd.DataFrame({
        'Feature'   : features,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)

    return importance_df, round(clf.score(X, y) * 100, 2)

# ── KMeans clustering ──────────────────────────────────────────────────────────
def cluster_transactions(selected_user, df, n_clusters=3):
    df = filter_user(df, selected_user).copy()
    X = df[['amount','hour','month_num']].fillna(0)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    return df[['pca1','pca2','cluster','amount','status']]

# ── wordcloud of receiver names ────────────────────────────────────────────────
def create_wordcloud(selected_user, df):
    df = filter_user(df, selected_user)
    text = ' '.join(df['receiver_name'].tolist() + df['sender_name'].tolist())
    wc = WordCloud(width=600, height=300, background_color='white',
                   colormap='plasma', max_words=100)
    wc.generate(text)
    return wc
