import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("clustered_customers.csv")

df = load_data()


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
.main{
    background-color:#f8f9fa;
}

h1,h2,h3{
    color:#0E4D92;
}

div[data-testid="metric-container"]{
    background:#ffffff;
    border-radius:12px;
    padding:15px;
    border:1px solid #dddddd;
    box-shadow:0 2px 5px rgba(0,0,0,0.1);
}

.sidebar .sidebar-content{
    background:#0E4D92;
}

footer{
    visibility:hidden;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("clustered_customers.csv")

df = load_data()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.image(
    "https://img.icons8.com/color/96/combo-chart--v1.png",
    width=90
)

st.sidebar.title("Navigation")

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

# =====================================================
# HOME PAGE
# =====================================================
if menu == "🏠 Home":

    st.title("📊 Customer Segmentation Dashboard")

    st.write("""
This dashboard performs **Customer Segmentation** using the
**K-Means Clustering Algorithm**.

The segmentation is based on:

- Age
- Annual Income
- Spending Score

Use the sidebar to explore different sections.
""")

    st.divider()

    total_customers = len(df)
    total_features = df.shape[1]
    total_clusters = df["Cluster"].nunique()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Customers", total_customers)
    c2.metric("Total Features", total_features)
    c3.metric("Total Clusters", total_clusters)

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# =====================================================
# DATA OVERVIEW
# =====================================================
elif menu == "📊 Data Overview":

    st.title("📊 Data Overview")

    st.subheader("Dataset")

    st.dataframe(df, use_container_width=True)

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
    st.dataframe(df.describe(), use_container_width=True)



# =====================================================
# EDA PAGE
# =====================================================
elif menu == "📈 EDA":

    st.title("📈 Exploratory Data Analysis")

    # -------------------------
    # Age Distribution
    # -------------------------
    st.subheader("Age Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        data=df,
        x="Age",
        bins=20,
        kde=True,
        color="royalblue",
        ax=ax
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    # -------------------------
    # Gender Distribution
    # -------------------------
    st.subheader("Gender Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        data=df,
        x="Gender",
        palette="Set2",
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Annual Income Distribution
    # -------------------------
    st.subheader("Annual Income Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        data=df,
        x="Annual Income (k$)",
        bins=20,
        kde=True,
        color="green",
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Spending Score Distribution
    # -------------------------
    st.subheader("Spending Score Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        data=df,
        x="Spending Score (1-100)",
        bins=20,
        kde=True,
        color="orange",
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Correlation Heatmap
    # -------------------------
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="Blues",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Boxplots
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Annual Income Boxplot")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.boxplot(
            y=df["Annual Income (k$)"],
            color="skyblue",
            ax=ax
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Spending Score Boxplot")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.boxplot(
            y=df["Spending Score (1-100)"],
            color="salmon",
            ax=ax
        )

        st.pyplot(fig)

    # -------------------------
    # Scatter Plot
    # -------------------------
    st.subheader("Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(9,6))

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

    # -------------------------
    # Pair Plot
    # -------------------------
    st.subheader("Pair Plot")

    pair_fig = sns.pairplot(
        df,
        vars=[
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ],
        hue="Cluster",
        palette="Set2"
    )

    st.pyplot(pair_fig)



# =====================================================
# CUSTOMER SEGMENTATION PAGE
# =====================================================
elif menu == "🤖 Customer Segmentation":

    st.title("🤖 Customer Segmentation")

    st.write("""
The dataset has been segmented using the **K-Means Clustering Algorithm**
based on customer Annual Income and Spending Score.
""")

    st.divider()

    # Cluster Count
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

    summary.columns = [
        "Average Age",
        "Average Income",
        "Average Spending Score"
    ]

    st.dataframe(summary, use_container_width=True)

    st.divider()

    # Average Metrics
    st.subheader("Average Customer Metrics by Cluster")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Age",
        round(df["Age"].mean(),1)
    )

    col2.metric(
        "Average Income",
        round(df["Annual Income (k$)"].mean(),1)
    )

    col3.metric(
        "Average Spending",
        round(df["Spending Score (1-100)"].mean(),1)
    )


# =====================================================
# CLUSTER VISUALIZATION
# =====================================================
elif menu == "📉 Cluster Visualization":

    st.title("📉 Cluster Visualization")

    st.subheader("Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(9,6))

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

    # Average Values
    st.subheader("Average Values by Cluster")

    avg = df.groupby("Cluster")[[
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]].mean()

    st.bar_chart(avg)

    st.divider()

    # Pie Chart
    st.subheader("Cluster Distribution")

    pie = df["Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6,6))

    ax.pie(
        pie,
        labels=pie.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Customer Distribution")

    st.pyplot(fig)

    st.divider()

    # Income by Cluster
    st.subheader("Income by Cluster")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.boxplot(
        data=df,
        x="Cluster",
        y="Annual Income (k$)",
        palette="Set3",
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    # Spending Score by Cluster
    st.subheader("Spending Score by Cluster")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.boxplot(
        data=df,
        x="Cluster",
        y="Spending Score (1-100)",
        palette="Pastel1",
        ax=ax
    )

    st.pyplot(fig)


# =====================================================
# BUSINESS INSIGHTS
# =====================================================
elif menu == "💼 Business Insights":

    st.title("💼 Business Insights")

    st.success("Customer Segmentation Completed Successfully!")

    st.divider()

    cluster = st.selectbox(
        "Select Customer Cluster",
        sorted(df["Cluster"].unique())
    )

    cluster_df = df[df["Cluster"] == cluster]

    st.subheader(f"Cluster {cluster} Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers",
        len(cluster_df)
    )

    col2.metric(
        "Average Income",
        round(cluster_df["Annual Income (k$)"].mean(), 2)
    )

    col3.metric(
        "Average Spending",
        round(cluster_df["Spending Score (1-100)"].mean(), 2)
    )

    st.divider()

    st.subheader("Average Age")

    st.metric(
        "Average Age",
        round(cluster_df["Age"].mean(), 2)
    )

    st.divider()

    st.subheader("Business Recommendation")

    if cluster == 0:

        st.info("""
### 🏆 Premium Customers

- High Income
- High Spending

**Recommendation**

- VIP Membership
- Loyalty Rewards
- Premium Products
- Exclusive Offers
- Personalized Marketing
""")

    elif cluster == 1:

        st.warning("""
### 💰 High Income - Low Spending

**Recommendation**

- Targeted Discounts
- Product Bundles
- Personalized Recommendations
- Email Marketing
- Festival Offers
""")

    elif cluster == 2:

        st.success("""
### 💵 Budget Customers

**Recommendation**

- Affordable Products
- Cashback Offers
- Combo Deals
- Coupons
- Seasonal Discounts
""")

    elif cluster == 3:

        st.info("""
### 🛍 Active Customers

**Recommendation**

- Reward Points
- Referral Program
- Mobile Notifications
- Membership Benefits
- Early Product Access
""")

    else:

        st.success("""
### 👥 Regular Customers

**Recommendation**

- Customer Engagement
- Feedback Program
- Birthday Offers
- Personalized Emails
- Loyalty Program
""")

    st.divider()

    st.subheader("Customers in Selected Cluster")

    st.dataframe(cluster_df, use_container_width=True)


# =====================================================
# DOWNLOAD PAGE
# =====================================================
elif menu == "📥 Download Data":

    st.title("📥 Download Dataset")

    st.write("Download the clustered customer dataset as CSV.")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
"""
<hr>

<div style='text-align:center;'>

### 📊 Customer Segmentation Dashboard

Developed by **Rohit Kumar Prajapati**

**Technology Used**

Python • Streamlit • Pandas • NumPy • Matplotlib • Seaborn • Scikit-learn

© 2026 All Rights Reserved

</div>
""",
unsafe_allow_html=True
)
