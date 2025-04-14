import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import pairwise_distances

# Set page configuration
st.set_page_config(page_title="Multi-Page Dashboard", page_icon="📊", layout="wide")

# Tạo điều khiển chọn trang từ sidebar
page = st.sidebar.radio("Chọn trang", 
                          ("Data Overview", "Product Recommendation", "Customer Segmentation", "Data Analysis"))

# ---------------------------
# Trang Data Overview
# ---------------------------
if page == "Data Overview":
    st.title("Data Overview")
    st.write("Hiển thị thông tin và thống kê của dữ liệu.")
    
    @st.cache_data
    def load_overview_data():
        transactions = pd.read_csv("Processed_transactions.csv")
        products = pd.read_csv("Products_with_Categories.csv")
        transactions.columns = transactions.columns.str.strip()
        products.columns = products.columns.str.strip()
        if "price" in products.columns and "product_price" not in products.columns:
            products.rename(columns={"price": "product_price"}, inplace=True)
        return transactions, products

    transactions_df, products_df = load_overview_data()
    st.write("### Transactions (first 5 rows):")
    st.dataframe(transactions_df.head())
    st.write("### Products (first 5 rows):")
    st.dataframe(products_df.head())

# ---------------------------
# Trang Product Recommendation
# ---------------------------
elif page == "Product Recommendation":
    st.title("Product Recommendation")
    st.write("Đề xuất sản phẩm cho khách hàng dựa trên lịch sử giao dịch.")
    
    @st.cache_data
    def load_recommendation_data():
        transactions = pd.read_csv("Processed_transactions.csv")
        products = pd.read_csv("Products_with_Categories.csv")
        transactions.columns = transactions.columns.str.strip()
        products.columns = products.columns.str.strip()
        # Rename nếu cần
        if "price" in products.columns and "product_price" not in products.columns:
            products.rename(columns={"price": "product_price"}, inplace=True)
        return transactions, products

    transactions_df, products_df = load_recommendation_data()
    
    # Chọn khách hàng (dựa trên Member_number)
    if "Member_number" not in transactions_df.columns:
        st.error("Dữ liệu giao dịch không có cột 'Member_number'")
    else:
        customer_list = transactions_df["Member_number"].unique().tolist()
        selected_customer = st.selectbox("Select a customer:", customer_list)
        st.write("Selected customer:", selected_customer)
        
        # Lấy giao dịch của khách hàng đã chọn
        customer_transactions = transactions_df[transactions_df["Member_number"] == selected_customer]
        st.write("Customer transactions (first 5 rows):")
        st.dataframe(customer_transactions.head())
        
        # Lấy danh sách productId khách hàng đã mua
        customer_product_ids = customer_transactions["productId"].unique()
        recommended_products = products_df[products_df["productId"].isin(customer_product_ids)]
        st.write("Products purchased by the customer:")
        st.dataframe(recommended_products[["productId", "productName", "Category", "product_price"]])
        
        # Sản phẩm liên quan
        st.subheader("Related Products")
        if not recommended_products.empty:
            selected_product = st.selectbox("Select a product to see related products:", recommended_products["productName"].unique())
            try:
                selected_category = products_df.loc[products_df["productName"] == selected_product, "Category"].values[0]
                related_products = products_df[products_df["Category"] == selected_category]
                st.write("Related products (same Category):")
                st.dataframe(related_products[["productId", "productName", "Category", "product_price"]])
            except Exception as e:
                st.error("Error retrieving related products: " + str(e))
        else:
            st.info("No recommended products available for the selected customer.")
            
        # Tìm kiếm sản phẩm
        st.subheader("Product Search")
        search_query = st.text_input("Enter a product keyword:")
        if search_query:
            search_results = products_df[products_df["productName"].str.contains(search_query, case=False, na=False)]
            st.write(f"Found {len(search_results)} products:")
            st.dataframe(search_results[["productId", "productName", "Category", "product_price"]])
        else:
            st.info("Please enter a product keyword to search.")

# ---------------------------
# Trang Customer Segmentation
# ---------------------------
elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.write("Phân đoạn khách hàng dựa trên dữ liệu RFM (Recency, Frequency, Monetary).")
    
    @st.cache_data
    def load_rfm_data():
        rfm_data = pd.read_csv("rfm.csv")
        return rfm_data

    rfm_data = load_rfm_data()

    # -------------------
    # Elbow Method for KMeans
    # -------------------
    st.subheader("KMeans Clustering - Elbow Method")
    
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])

    # Elbow method to determine the optimal number of clusters
    k_range = range(1, 11)
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_scaled)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertia, marker='o')
    ax.set_title('Elbow Method for Optimal Clusters')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    st.pyplot(fig)

    # Based on the elbow plot, choose the number of clusters
    optimal_kmeans_clusters = 3
    st.write(f"Optimal number of clusters determined from the elbow method: {optimal_kmeans_clusters}")
    
    # KMeans clustering with optimal clusters
    kmeans = KMeans(n_clusters=optimal_kmeans_clusters, random_state=42)
    rfm_data['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Show the clustered data
    st.write("### Clustered RFM Data (KMeans):")
    st.dataframe(rfm_data.head())

    # KMeans Visualization
    st.subheader("KMeans Cluster Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm_data, x='Recency', y='Monetary', hue='KMeans_Cluster', palette='viridis', ax=ax)
    st.pyplot(fig)

    # -------------------
    # Hierarchical Clustering
    # -------------------
    st.subheader("Hierarchical Clustering - Dendrogram")
    
    # Create a linkage matrix for hierarchical clustering
    Z = linkage(rfm_scaled, method='ward')  # 'ward' minimizes variance within clusters
    
    # Plotting the Dendrogram
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(Z, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Distance')
    st.pyplot(fig)

    # Based on the dendrogram, we can cut the tree to form clusters
    hierarchical_clusters = fcluster(Z, t=optimal_kmeans_clusters, criterion='maxclust')
    rfm_data['Hierarchical_Cluster'] = hierarchical_clusters

    # Show the clustered data
    st.write("### Clustered RFM Data (Hierarchical):")
    st.dataframe(rfm_data.head())

    # Hierarchical Clustering Visualization
    st.subheader("Hierarchical Clustering Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm_data, x='Recency', y='Monetary', hue='Hierarchical_Cluster', palette='Set2', ax=ax)
    st.pyplot(fig)

    # Cluster Summary for both methods
    st.subheader("Cluster Summary")
    st.write("### KMeans Cluster Summary")
    st.write(rfm_data.groupby('KMeans_Cluster').mean())

    st.write("### Hierarchical Cluster Summary")
    st.write(rfm_data.groupby('Hierarchical_Cluster').mean())

# ---------------------------
# Trang Data Analysis
# ---------------------------
elif page == "Data Analysis":
    st.title("Data Analysis")
    st.write("Thực hiện phân tích dữ liệu cho dữ liệu RFM và các giao dịch.")

    # Load RFM data for analysis
    rfm_data = pd.read_csv("rfm.csv")

    # Clean data by dropping rows with missing values in the RFM columns
    rfm_data = rfm_data.dropna(subset=['Recency', 'Frequency', 'Monetary'])

    # Ensure that all columns are numeric
    rfm_data[['Recency', 'Frequency', 'Monetary']] = rfm_data[['Recency', 'Frequency', 'Monetary']].apply(pd.to_numeric, errors='coerce')

    # Summary statistics
    st.subheader("Descriptive Statistics")
    st.write(rfm_data.describe())

    # Visualize distributions of RFM features
    st.subheader("RFM Feature Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.histplot(rfm_data['Recency'], bins=20, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution of Recency')

    sns.histplot(rfm_data['Frequency'], bins=20, kde=True, ax=axes[1], color='orange')
    axes[1].set_title('Distribution of Frequency')

    sns.histplot(rfm_data['Monetary'], bins=20, kde=True, ax=axes[2], color='green')
    axes[2].set_title('Distribution of Monetary')

    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Ứng Dụng Phân Cụm Khách Hàng</p>
    <p>Nguồn dữ liệu: Dữ liệu giao dịch thuộc chuỗi cửa hàng tiện lợi ở Mỹ • Ngày: 01-2014 -> 12-2015</p>
</div>
""", unsafe_allow_html=True)
