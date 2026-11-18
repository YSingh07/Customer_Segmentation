
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Step 1: Load customer data with cluster assignments
df = pd.read_csv("data/customer_segments.csv")

# Step 2: Define the features used for clustering
feature_cols = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

# Step 3: Calculate the number of customers in each cluster
print("Number of customers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Step 4: Calculate average customer characteristics
cluster_summary = df.groupby("Cluster")[feature_cols].mean().round(2)

print("\nAverage characteristics of each cluster:")
print(cluster_summary)

# Step 5: Calculate additional statistics
cluster_stats = df.groupby("Cluster").agg(
    Customer_Count=("Cluster", "count"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2)

print("\nDetailed cluster statistics:")
print(cluster_stats)

# Step 6: Save the summary to a CSV file
cluster_stats.to_csv("data/cluster_summary.csv")

# Step 7: Load the trained model and scaler
kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Step 8: Convert cluster centers back to original units
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Step 9: Create the cluster visualization
plt.figure(figsize=(10, 6))

for cluster_id in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster_id]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {cluster_id}",
        alpha=0.7
    )

# Step 10: Display the center of each cluster
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    label="Cluster Centers"
)

# Step 11: Add chart labels and formatting
plt.title("Customer Segmentation Using K-Means (K = 5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Step 12: Save and display the chart
plt.savefig("data/cluster_visualization.png")
plt.show()

print("\nAnalysis completed successfully!")
print("Summary saved to data/cluster_summary.csv")
print("Chart saved to data/cluster_visualization.png")
