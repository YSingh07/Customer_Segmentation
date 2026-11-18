
# 1. Import the required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 2. Load the original dataset
df = pd.read_csv("data/Mall_Customers.csv")

# 3. Display the original dataset shape
print("Original dataset shape:", df.shape)

# 4. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Remove completely duplicated rows
df = df.drop_duplicates()

# 6. Select the features for clustering
features = df[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].copy()

# 7. Check whether the selected features are numeric
print("\nSelected feature data types:")
print(features.dtypes)

# 8. Handle missing values in the selected features
features = features.dropna()

# 9. Create a scaler object
scaler = StandardScaler()

# 10. Scale the selected features
scaled_features = scaler.fit_transform(features)

# 11. Convert scaled data back to a DataFrame
scaled_df = pd.DataFrame(
    scaled_features,
    columns=features.columns,
    index=features.index
)

# 12. Display the prepared data
print("\nFirst five rows of original features:")
print(features.head())

print("\nFirst five rows of scaled features:")
print(scaled_df.head())

# 13. Check the final dataset shape
print("\nPrepared dataset shape:", scaled_df.shape)

# 14. Save the prepared data
scaled_df.to_csv("data/scaled_customers.csv", index=False)

print("\nPreprocessing completed successfully!")
