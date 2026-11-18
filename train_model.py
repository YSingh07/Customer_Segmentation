
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Step 2: Select features
features = df[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].copy()

# Step 3: Remove missing feature values
features = features.dropna()

# Step 4: Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Calculate inertia for different K values
inertias = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_features)
    inertias.append(model.inertia_)

# Step 6: Display the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias, marker="o")
plt.title("Elbow Method for Customer Segmentation")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.savefig("data/elbow_curve.png")
plt.show()

# Step 7: Train the final K-Means model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(scaled_features)

# Step 8: Create the models folder if needed
os.makedirs("models", exist_ok=True)

# Step 9: Save the trained model and scaler
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Step 10: Predict cluster assignments for existing customers
df = df.loc[features.index].copy()
df["Cluster"] = kmeans.labels_

# Step 11: Save customer segment assignments
df.to_csv("data/customer_segments.csv", index=False)

print("\nModel training completed!")
print("Number of clusters:", kmeans.n_clusters)
print("Customers in each cluster:")
print(df["Cluster"].value_counts().sort_index())
print("\nSaved model: models/kmeans_model.pkl")
print("Saved scaler: models/scaler.pkl")
print("Saved customer assignments: data/customer_segments.csv")
