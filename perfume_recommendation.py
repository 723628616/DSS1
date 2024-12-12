import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.rename(columns={"type": "scent_type", "price": "price", "brand": "brand", "title": "title"})
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    return data

# Normalize weights (total sum = 1)
def normalize_weights(raw_weights):
    total = sum(raw_weights.values())
    return {key: value / total for key, value in raw_weights.items()}

# Generate utility curves based on normalized weights
def generate_utility_curves(normalized_weights):
    curve_params = {
        "Budget Utility": {"a": normalized_weights["Budget"] * 0.02, "b": normalized_weights["Budget"] * 0.1, "c": 0},
        "Scent Utility": {"a": normalized_weights["Scent"] * 0.01, "b": normalized_weights["Scent"] * 0.2, "c": 0},
        "Brand Utility": {"a": normalized_weights["Brand"] * 0.03, "b": normalized_weights["Brand"] * 0.05, "c": 0},
    }
    x = np.linspace(0, 10, 100)
    curves = []
    for label, params in curve_params.items():
        a, b, c = params["a"], params["b"], params["c"]
        y = a * x**2 + b * x + c
        curves.append((label, x, y, f"y = {a:.2f}x² + {b:.2f}x + {c:.2f}"))
    return curves

# Streamlit app
def main():
    st.title("Customized Perfume Recommendation System with Normalized Weights")

    # Upload CSV
    uploaded_file = st.file_uploader("Upload perfume data (CSV file)", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to continue.")
        return

    # Load data
    data = load_data(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # User filters
    st.sidebar.header("Filter Preferences")
    selected_scent = st.sidebar.multiselect(
        "Select scent type:", options=data["scent_type"].unique(), default=data["scent_type"].unique()
    )
    selected_budget = st.sidebar.slider(
        "Select price range:",
        min_value=float(data["price"].min()),
        max_value=float(data["price"].max()),
        value=(float(data["price"].min()), float(data["price"].max()))
    )
    selected_brand = st.sidebar.multiselect(
        "Select brands:", options=data["brand"].unique(), default=data["brand"].unique()
    )

    # User-defined weights
    st.sidebar.header("Set Factor Weights Dynamically")
    budget_weight = st.sidebar.slider("Budget weight", min_value=1, max_value=10, value=3)
    scent_weight = st.sidebar.slider("Scent weight", min_value=1, max_value=10, value=5)
    brand_weight = st.sidebar.slider("Brand weight", min_value=1, max_value=10, value=2)

    # Normalize weights
    raw_weights = {"Budget": budget_weight, "Scent": scent_weight, "Brand": brand_weight}
    normalized_weights = normalize_weights(raw_weights)
    st.sidebar.write("Normalized Weights:", normalized_weights)

    # Filter data
    filtered_data = data[
        (data["scent_type"].isin(selected_scent)) &
        (data["price"] >= selected_budget[0]) &
        (data["price"] <= selected_budget[1]) &
        (data["brand"].isin(selected_brand))
    ]

    if filtered_data.empty:
        st.warning("No results match your filters. Please adjust your preferences.")
        return

    # Generate and plot utility curves
    st.subheader("Utility Functions Based on Normalized Weights")
    st.write("These utility functions are dynamically generated based on normalized user-defined weights (total sum = 1).")

    curves = generate_utility_curves(normalized_weights)
    fig = go.Figure()
    for label, x, y, equation in curves:
        # Add utility curve to plot
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label, hovertext=equation))

        # Add annotation for maximum utility point
        max_index = np.argmax(y)
        max_x, max_y = x[max_index], y[max_index]
        fig.add_annotation(
            x=max_x, y=max_y,
            text=f"Max: {max_y:.2f}",
            showarrow=True, arrowhead=2, ax=20, ay=-30,
            font=dict(size=10)
        )

    fig.update_layout(
        title="Utility Functions for Budget, Scent, and Brand (Normalized Weights)",
        xaxis_title="Value",
        yaxis_title="Utility",
        legend_title="Factors",
        hovermode="x unified"
    )
    st.plotly_chart(fig)

    # Utility function statistics
    st.subheader("Utility Function Statistics")
    stats = []
    for label, x, y, equation in curves:
        stats.append({
            "Factor": label,
            "Equation": equation,
            "Max": np.max(y),
            "Min": np.min(y),
            "Average": np.mean(y)
        })
    stats_df = pd.DataFrame(stats)
    st.dataframe(stats_df)

    # Calculate utility scores
    def calculate_score(row, weights, selected_budget, selected_scent, selected_brand):
        budget_utility = weights["Budget"] * (row["price"] - selected_budget[0]) / (selected_budget[1] - selected_budget[0])
        scent_utility = weights["Scent"] if row["scent_type"] in selected_scent else 0
        brand_utility = weights["Brand"] if row["brand"] in selected_brand else 0
        return budget_utility + scent_utility + brand_utility

    filtered_data["Utility_Score"] = filtered_data.apply(
        lambda row: calculate_score(row, normalized_weights, selected_budget, selected_scent, selected_brand),
        axis=1
    )
    filtered_data = filtered_data.sort_values(by="Utility_Score", ascending=False)

    # Display recommendation results
    st.subheader("Recommendation Results")
    st.dataframe(filtered_data[["title", "price", "scent_type", "brand", "Utility_Score"]])

if __name__ == "__main__":
    main()
