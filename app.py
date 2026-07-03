import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import helper
import preprocessor

st.sidebar.title("💸 UPI Transaction Analyser")
st.sidebar.markdown("Upload your UPI transaction CSV to get started!")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocessor.preprocess(df)

    # ── User selection ──────────────────────────────────────────────────────────
    all_users = sorted(set(df['sender_name'].unique().tolist() + df['receiver_name'].unique().tolist()))
    all_users.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Select User", all_users)

    if st.sidebar.button("Show Analysis"):

        # ── TOP STATISTICS ──────────────────────────────────────────────────────
        total_tx, total_amount, success, failed, avg_amount = helper.fetch_stats(selected_user, df)
        st.title("📊 Top Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.header("Total Txns")
            st.title(total_tx)
        with col2:
            st.header("Total Amount")
            st.title(f"₹{total_amount:,}")
        with col3:
            st.header("Success")
            st.title(success)
        with col4:
            st.header("Failed")
            st.title(failed)
        with col5:
            st.header("Avg Amount")
            st.title(f"₹{avg_amount}")

        # ── SUCCESS VS FAILED ───────────────────────────────────────────────────
        st.title("✅ Success vs Failed")
        status_data = helper.status_breakdown(selected_user, df)
        fig, ax = plt.subplots()
        ax.pie(status_data.values, labels=status_data.index,
               autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'])
        st.pyplot(fig)

        # ── MONTHLY TIMELINE ────────────────────────────────────────────────────
        st.title("📅 Monthly Transaction Volume (₹)")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(timeline['time'], timeline['amount'], color='green', marker='o')
        plt.xticks(rotation='vertical')
        ax.set_ylabel("Amount (₹)")
        st.pyplot(fig)

        # ── DAILY TIMELINE ──────────────────────────────────────────────────────
        st.title("📆 Daily Transaction Volume (₹)")
        daily = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(daily['date_only'], daily['amount'], color='black')
        plt.xticks(rotation='vertical')
        ax.set_ylabel("Amount (₹)")
        st.pyplot(fig)

        # ── ACTIVITY MAP ────────────────────────────────────────────────────────
        st.title("🗺️ Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Busiest Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            ax.set_ylabel("Transactions")
            st.pyplot(fig)

        with col2:
            st.header("Busiest Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            ax.set_ylabel("Transactions")
            st.pyplot(fig)

        # ── HOURLY ACTIVITY ─────────────────────────────────────────────────────
        st.title("🕐 Hourly Transaction Pattern")
        hourly = helper.hourly_activity(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(hourly.index, hourly.values, color='teal')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Amount (₹)")
        st.pyplot(fig)

        # ── AMOUNT DISTRIBUTION ─────────────────────────────────────────────────
        st.title("💰 Amount Distribution")
        bins_data = helper.amount_bins(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(bins_data.index, bins_data.values, color='steelblue')
        ax.set_ylabel("Number of Transactions")
        ax.set_xlabel("Amount Range (₹)")
        st.pyplot(fig)

        # ── CATEGORY ANALYSIS ───────────────────────────────────────────────────
        st.title("🏷️ Spending by Category")
        cat_data = helper.category_analysis(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(cat_data.index, cat_data.values, color='coral')
        ax.set_xlabel("Total Amount (₹)")
        st.pyplot(fig)

        # ── BANK DISTRIBUTION ───────────────────────────────────────────────────
        st.title("🏦 UPI Bank Distribution")
        bank_data = helper.bank_distribution(selected_user, df)
        fig, ax = plt.subplots()
        ax.pie(bank_data.values, labels=bank_data.index, autopct='%1.1f%%')
        st.pyplot(fig)

        # ── TOP SENDERS & RECEIVERS ─────────────────────────────────────────────
        if selected_user == 'Overall':
            st.title("👤 Top Senders & Receivers")
            col1, col2 = st.columns(2)

            with col1:
                st.header("Top Senders")
                top_s = helper.top_senders(df)
                fig, ax = plt.subplots()
                ax.bar(top_s.index, top_s.values, color='#3498db')
                plt.xticks(rotation='vertical')
                ax.set_ylabel("Total Amount (₹)")
                st.pyplot(fig)

            with col2:
                st.header("Top Receivers")
                top_r = helper.top_receivers(df)
                fig, ax = plt.subplots()
                ax.bar(top_r.index, top_r.values, color='#e67e22')
                plt.xticks(rotation='vertical')
                ax.set_ylabel("Total Amount (₹)")
                st.pyplot(fig)

        # ── WORDCLOUD ───────────────────────────────────────────────────────────
        st.title("☁️ WordCloud — Active Users")
        wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # ── ANOMALY DETECTION ───────────────────────────────────────────────────
        st.title("🚨 Anomaly Detection (Suspicious Transactions)")
        anomalies = helper.detect_anomalies(selected_user, df)
        st.dataframe(anomalies)

        # ── RANDOM FOREST FEATURE IMPORTANCE ───────────────────────────────────
        st.title("🤖 ML — What Predicts Transaction Success?")
        importance_df, accuracy = predict_success_wrapper(df)
        st.markdown(f"**Model Accuracy: {accuracy}%**")
        fig, ax = plt.subplots()
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='darkgreen')
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)

        # ── KMEANS CLUSTERING ───────────────────────────────────────────────────
        st.title("🔵 Transaction Clusters (KMeans + PCA)")
        cluster_df = helper.cluster_transactions(selected_user, df)
        fig, ax = plt.subplots()
        scatter = ax.scatter(cluster_df['pca1'], cluster_df['pca2'],
                             c=cluster_df['cluster'], cmap='Set1', alpha=0.5, s=10)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        st.pyplot(fig)

else:
    st.title("💸 UPI Transaction Analyser")
    st.markdown("""
    ### How to use:
    1. Upload your UPI transaction CSV from the sidebar
    2. Select a user or keep **Overall**
    3. Click **Show Analysis**

    ### CSV Format Required:
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

    > **Don't have data?** Download our sample dataset below 👇
    """)

    # Generate and offer sample dataset download
    import subprocess
    import os
    if not os.path.exists("upi_transactions.csv"):
        subprocess.run(["python", "generate_data.py"])

    with open("upi_transactions.csv", "rb") as f:
        st.download_button(
            label="📥 Download Sample Dataset",
            data=f,
            file_name="upi_transactions.csv",
            mime="text/csv"
        )


def predict_success_wrapper(df):
    return helper.predict_success(df)
