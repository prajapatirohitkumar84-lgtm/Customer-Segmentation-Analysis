import streamlit as st
import pandas as pd

df = pd.read_csv(r"D:\Project (Navodita infotech)\Customer Segmentation Analysis-[Rohit Kumar Prajapati]\train.csv")

st.title("Customer Segmentation Dashboard")
st.dataframe(df)


import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("clustered_customers.csv")

df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📂 Navigation")

menu = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Data Overview",
        "📈 EDA",
        "🤖 Customer Segmentation",
        "📉 Cluster Visualization",
        "💼 Business Insights",
        "📥 Download Data"
    ]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if menu == "🏠 Home":

    st.title("📊 Customer Segmentation Dashboard")

    st.markdown("""
    This dashboard analyzes customer behavior using
    **K-Means Clustering** to identify different customer segments
    based on Annual Income and Spending Score.
    """)

    st.divider()

    total_customers = len(df)
    total_features = df.shape[1]
    total_clusters = df["Cluster"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", total_customers)
    col2.metric("Total Features", total_features)
    col3.metric("Total Clusters", total_clusters)

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

# -----------------------------
# DATA OVERVIEW
# -----------------------------
elif menu == "📊 Data Overview":

    st.title("📊 Data Overview")

    st.subheader("Dataset")

    st.dataframe(df)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Shape")
        st.write(df.shape)

    with col2:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

    st.divider()

    st.subheader("Data Types")

    st.write(df.dtypes)

    st.divider()

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# EDA PAGE
# -----------------------------
elif menu == "📈 EDA":

    st.title("📈 Exploratory Data Analysis")

    # -----------------------------
    # Age Distribution
    # -----------------------------
    st.subheader("Age Distribution")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    # -----------------------------
    # Gender Distribution
    # -----------------------------
    st.subheader("Gender Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="Gender", ax=ax)

    st.pyplot(fig)

    # -----------------------------
    # Annual Income Distribution
    # -----------------------------
    st.subheader("Annual Income Distribution")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["Annual Income (k$)"], bins=20, kde=True, ax=ax)

    st.pyplot(fig)

    # -----------------------------
    # Spending Score Distribution
    # -----------------------------
    st.subheader("Spending Score Distribution")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True, ax=ax)

    st.pyplot(fig)

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(7,5))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    # -----------------------------
    # Box Plot
    # -----------------------------
    st.subheader("Annual Income Boxplot")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(
        data=df,
        y="Annual Income (k$)",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Spending Score Boxplot")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(
        data=df,
        y="Spending Score (1-100)",
        ax=ax
    )

    st.pyplot(fig)

    # -----------------------------
    # Scatter Plot
    # -----------------------------
    st.subheader("Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="Set2",
        ax=ax
    )

    st.pyplot(fig)


# -----------------------------
# CUSTOMER SEGMENTATION PAGE
# -----------------------------
elif menu == "🤖 Customer Segmentation":

    st.title("🤖 Customer Segmentation")

    st.write("""
    The customers have been segmented using the **K-Means Clustering**
    algorithm based on **Annual Income** and **Spending Score**.
    """)

    st.divider()

    # Cluster-wise Customer Count
    st.subheader("Cluster-wise Customer Count")

    cluster_count = (
        df["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_count.columns = ["Cluster", "Customers"]

    st.dataframe(cluster_count, use_container_width=True)

    # Bar Chart
    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        data=df,
        x="Cluster",
        palette="Set2",
        ax=ax
    )

    ax.set_title("Customers in Each Cluster")

    st.pyplot(fig)

    st.divider()

    # Cluster Summary
    st.subheader("Cluster Summary")

    summary = df.groupby("Cluster").agg({

        "Age":"mean",
        "Annual Income (k$)":"mean",
        "Spending Score (1-100)":"mean"

    }).round(2)

    st.dataframe(summary, use_container_width=True)

# -----------------------------
# CLUSTER VISUALIZATION
# -----------------------------
elif menu == "📉 Cluster Visualization":

    st.title("📉 Cluster Visualization")

    st.subheader("Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="Set2",
        s=120,
        ax=ax
    )

    ax.set_title("Customer Segments")

    st.pyplot(fig)

    st.divider()

    st.subheader("Average Values by Cluster")

    avg = df.groupby("Cluster")[[
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]].mean()

    st.bar_chart(avg)

    st.divider()

    st.subheader("Customers in Each Cluster")

    pie = df["Cluster"].value_counts()

    fig, ax = plt.subplots(figsize=(6,6))

    ax.pie(
        pie,
        labels=pie.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Cluster Distribution")

    st.pyplot(fig)



# -----------------------------
# BUSINESS INSIGHTS
# -----------------------------
elif menu == "💼 Business Insights":

    st.title("💼 Business Insights")

    st.success("Customer Segments Generated Successfully!")

    st.divider()

    cluster = st.selectbox(
        "Select Cluster",
        sorted(df["Cluster"].unique())
    )

    st.subheader(f"Cluster {cluster}")

    cluster_df = df[df["Cluster"] == cluster]

    st.metric("Customers", len(cluster_df))

    st.write("### Average Values")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Age",
        round(cluster_df["Age"].mean(),1)
    )

    col2.metric(
        "Average Income",
        round(cluster_df["Annual Income (k$)"].mean(),1)
    )

    col3.metric(
        "Average Spending",
        round(cluster_df["Spending Score (1-100)"].mean(),1)
    )

    st.divider()

    # Business Recommendation

    if cluster == 0:

        st.info("""
### Premium Customers

✔ High Income

✔ High Spending

### Recommendation

• VIP Membership

• Premium Products

• Loyalty Rewards

• Personalized Offers

• Early Product Access
""")

    elif cluster == 1:

        st.warning("""
### High Income - Low Spending

### Recommendation

• Personalized Discounts

• Product Recommendation

• Premium Membership

• Email Marketing

• Special Coupons
""")

    elif cluster == 2:

        st.success("""
### Budget Customers

### Recommendation

• Affordable Products

• Combo Offers

• Festival Discounts

• Cashback

• Coupons
""")

    elif cluster == 3:

        st.info("""
### Active Shoppers

### Recommendation

• Loyalty Program

• Referral Rewards

• Personalized Suggestions

• Mobile Notifications

• Reward Points
""")

    else:

        st.success("""
### Regular Customers

### Recommendation

• Customer Engagement

• Membership Benefits

• Email Campaign

• Birthday Offers

• Customer Feedback Program
""")

    st.divider()

    st.subheader("Customers in Selected Cluster")

    st.dataframe(cluster_df)


# -----------------------------
# DOWNLOAD PAGE
# -----------------------------
elif menu == "📥 Download Data":

    st.title("📥 Download Clustered Dataset")

    st.write(
        "Download the clustered customer dataset."
    )

    csv = df.to_csv(index=False)

    st.download_button(

        label="⬇ Download CSV",

        data=csv,

        file_name="clustered_customers.csv",

        mime="text/csv"

    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head())


# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.markdown(
"""
<center>

Developed by **Rohit Kumar Prajapati**

Customer Segmentation using K-Means Clustering

Streamlit Dashboard

</center>
""",
unsafe_allow_html=True
)
