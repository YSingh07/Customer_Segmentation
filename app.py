
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load customers and their existing cluster assignments
customers = pd.read_csv("data/customer_segments.csv")

FEATURES = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]



# Calculate the average income and spending for each cluster
cluster_profiles = customers.groupby("Cluster")[FEATURES].mean()

# Calculate the overall customer averages
overall_income = customers[FEATURES[0]].mean()
overall_spending = customers[FEATURES[1]].mean()


def get_segment_name(cluster_id):
    # Get the average values for this cluster
    profile = cluster_profiles.loc[cluster_id]

    income = profile[FEATURES[0]]
    spending = profile[FEATURES[1]]

    # Compare this cluster with the overall averages
    if income >= overall_income:
        income_label = "High Income"
    else:
        income_label = "Low Income"

    if spending >= overall_spending:
        spending_label = "High Spending"
    else:
        spending_label = "Low Spending"

    return f"{income_label} - {spending_label}"



# Route 1: Predict a customer's segment
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            income = float(request.form["income"])
            spending = float(request.form["spending"])

            if not 0 <= income <= 1000:
                raise ValueError(
                    "Income must be between 0 and 1000 thousand dollars."
                )

            if not 1 <= spending <= 100:
                raise ValueError(
                    "Spending score must be between 1 and 100."
                )

            input_data = pd.DataFrame(
                [[income, spending]],
                columns=FEATURES
            )

            scaled_data = scaler.transform(input_data)
            cluster = int(kmeans.predict(scaled_data)[0])

            prediction = {
                "cluster": cluster,
                "segment_name": get_segment_name(cluster),
                "income": income,
                "spending": spending
            }

        except ValueError as e:
            error = str(e) or "Please enter valid numeric values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


# Route 2: Display the dashboard
@app.route("/dashboard")
def dashboard():

    # Calculate total customers
    total_customers = len(customers)

    # Count the number of clusters
    num_clusters = customers["Cluster"].nunique()

    # Calculate overall averages
    avg_income = customers[FEATURES[0]].mean()
    avg_spending = customers[FEATURES[1]].mean()

    # Count customers in each cluster
    cluster_counts = (
        customers["Cluster"]
        .value_counts()
        .sort_index()
    )

    # Calculate characteristics of each cluster
    summary = customers.groupby("Cluster").agg(
        Customer_Count=("Cluster", "count"),
        Average_Income=(FEATURES[0], "mean"),
        Average_Spending=(FEATURES[1], "mean")
    ).round(2).reset_index()
    
    summary["Segment_Name"] = summary["Cluster"].apply(
        get_segment_name
    )

    # Prepare chart data
    chart_labels = cluster_counts.index.astype(str).tolist()
    chart_values = cluster_counts.tolist()

    return render_template(
        "dashboard.html",
        total_customers=total_customers,
        num_clusters=num_clusters,
        avg_income=avg_income,
        avg_spending=avg_spending,
        summary=summary.to_dict(orient="records"),
        chart_labels=chart_labels,
        chart_values=chart_values
    )


if __name__ == "__main__":
    app.run(debug=True)
