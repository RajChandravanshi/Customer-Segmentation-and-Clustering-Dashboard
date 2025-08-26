# Customer-Segmentation-and-Clustering-Dashboard

An **interactive dashboard** built with **Streamlit** for performing **customer segmentation** using machine learning.  
This app allows you to upload your dataset, apply clustering (currently **KMeans**), and explore insights through **visualizations** such as PCA projections, distribution plots, elbow method curves, and pair plots.  
It’s a handy tool for businesses and analysts to understand customer behavior, segment customers, and create **data-driven targeted marketing strategies**.

---

## Features

- **Upload CSV datasets** directly from the sidebar UI  
- **KMeans clustering** with adjustable number of clusters (k)  
- **Automatic preprocessing** – numeric feature selection & scaling using `StandardScaler`  
- **Cluster labeling** – clusters are automatically assigned unique labels for easy interpretation  
- **Rich visualizations** included:
  - **PCA 2D projection** – visualize clusters in a 2D plane  
  - **PCA 3D projection** – interactive view of clusters in 3D  
  - **Cluster distribution** – Pie chart & Bar chart for comparing cluster sizes  
  - **Elbow Method curve** – find the optimal number of clusters for KMeans  
  - **Pair Plot of features** – explore pairwise feature relationships across clusters  
- **Cluster statistics** – view cluster counts and proportions directly in the dashboard  
- **Download processed dataset** with cluster labels for further use in analysis or reporting  
- **Silhouette Score metric** – quick evaluation of clustering performance  
- **Simple, clean, and interactive UI** powered by **Streamlit**  
- **Scalable & modular codebase** – easy to extend with new clustering methods like DBSCAN, Agglomerative, or Gaussian Mixture Models (GMM)  

---

✨ With this tool, you can:  
- Identify **loyal customers vs. occasional buyers**  
- Segment customers based on **purchase patterns, demographics, or behavior**  
- Generate insights to design **personalized marketing campaigns**  
- Understand **customer churn risks** and optimize **customer lifetime value (CLV)**  
