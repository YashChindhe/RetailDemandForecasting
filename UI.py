import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression  # Example model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load the Data (Replace with your actual data loading)
# Assuming you have a CSV file or a Pandas DataFrame
# st.cache_data decorator is used to cache the data, so it's only loaded once.
@st.cache_data
def load_data():
    file = pd.read_csv('retail_store_inventory.csv')
    return prepare_data(file)

# 2. Prepare Data: Feature Engineering
def prepare_data(file):
    file.Date = file.Date.astype('datetime64[ns]')

    storeid_mapping = {'S001': 1, 'S002': 2, 'S003': 3, 'S004': 4, 'S005': 5}
    file['Store ID'].replace(storeid_mapping, inplace=True)

    productid_mapping = {'P0001': 1, 'P0002': 2, 'P0003': 3, 'P0004': 4, 'P0005': 5, 'P0006': 6, 'P0007': 7,
                         'P0008': 8, 'P0009': 9, 'P0010': 10, 'P0011': 11, 'P0012': 12, 'P0013': 13, 'P0014': 14,
                         'P0015': 15, 'P0016': 16, 'P0017': 17, 'P0018': 18, 'P0019': 19, 'P0020': 20}
    file['Product ID'].replace(productid_mapping, inplace=True)

    category_mapping = {'Groceries': 1, 'Toys': 2, 'Electronics': 3, 'Furniture': 4, 'Clothing': 5}
    file.Category.replace(category_mapping, inplace=True)

    region_mapping = {'North': 1, 'South': 2, 'West': 3, 'East': 4}
    file.Region.replace(region_mapping, inplace=True)

    weather_mapping = {'Rainy': 1, 'Sunny': 2, 'Cloudy': 3, 'Snowy': 4}
    file['Weather Condition'].replace(weather_mapping, inplace=True)

    seasonality_mapping = {'Autumn': 1, 'Summer': 2, 'Winter': 3, 'Spring': 4}
    file.Seasonality.replace(seasonality_mapping, inplace=True)

    file['Month'] = file['Date'].dt.month
    file['Year'] = file['Date'].dt.year
    file['Weekday'] = file['Date'].dt.day_name()
    return file

# 3. Streamlit App
def main():
    st.title("Demand Forecasting Dashboard")

    with st.expander("ℹ️ View Data Mappings"):
        st.subheader("Store ID Mapping")
        st.table(pd.DataFrame({
            'Store ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
            'Mapped Value': [1, 2, 3, 4, 5]
        }))

        st.subheader("Product ID Mapping")
        st.table(pd.DataFrame({
            'Product ID': [f'P{str(i).zfill(4)}' for i in range(1, 21)],
            'Mapped Value': list(range(1, 21))
        }))

        st.subheader("Category Mapping")
        st.table(pd.DataFrame({
            'Category': ['Groceries', 'Toys', 'Electronics', 'Furniture', 'Clothing'],
            'Mapped Value': [1, 2, 3, 4, 5]
        }))

        st.subheader("Region Mapping")
        st.table(pd.DataFrame({
            'Region': ['North', 'South', 'West', 'East'],
            'Mapped Value': [1, 2, 3, 4]
        }))

        st.subheader("Weather Condition Mapping")
        st.table(pd.DataFrame({
            'Weather Condition': ['Rainy', 'Sunny', 'Cloudy', 'Snowy'],
            'Mapped Value': [1, 2, 3, 4]
        }))

        st.subheader("Seasonality Mapping")
        st.table(pd.DataFrame({
            'Seasonality': ['Autumn', 'Summer', 'Winter', 'Spring'],
            'Mapped Value': [1, 2, 3, 4]
        }))


    # 4. Load and Prepare Data
    df = load_data()

    # 5. Add Filters
    product_filter = st.sidebar.multiselect("Select Product ID", df['Product ID'].unique(),
                                            default=df['Product ID'].unique())
    store_filter = st.sidebar.multiselect("Select Store ID", df['Store ID'].unique(), default=df['Store ID'].unique())
    category_filter = st.sidebar.multiselect("Select Category", df['Category'].unique(),
                                               default=df['Category'].unique())
    region_filter = st.sidebar.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())
    # Date filter should be a tuple, and use the min and max dates from the dataframe
    min_date = df['Date'].min().date()  # Extract date part
    max_date = df['Date'].max().date()  # Extract date part
    date_filter = st.sidebar.date_input("Select Date Range", value=(min_date, max_date))

    # 6. Apply Filters
    df_filtered = df[
        (df['Product ID'].isin(product_filter)) &
        (df['Store ID'].isin(store_filter)) &
        (df['Category'].isin(category_filter)) &
        (df['Region'].isin(region_filter)) &
        (df['Date'].dt.date >= date_filter[0]) &  # Compare dates, not datetimes
        (df['Date'].dt.date <= date_filter[1])  # Compare dates, not datetimes
        ]

    # 7. Group and Aggregate Data
    # Function to aggregate data, now includes actual_demand
    def aggregate_data(df, group_by_cols):
        if not group_by_cols:  # Handle the case where no columns are selected.
            return pd.DataFrame({'Demand Forecast': [df['Demand Forecast'].sum()],
                                 'actual_demand': [df['Units Sold'].sum()]})  # return sum of forecast and actual demand
        return df.groupby(group_by_cols).agg(
            demand_forecast=('Demand Forecast', 'sum'),
            actual_demand=('Units Sold', 'sum')  # sum of actual demand
        ).reset_index()

    # 8. Select Grouping Columns
    group_by_options = ['Date', 'Product ID', 'Store ID', 'Category', 'Region', 'Month', 'Year', 'Weekday']
    selected_group_by = st.multiselect("Group by", group_by_options, default=['Date'])

    # Ensure at least one grouping column is selected.
    if not selected_group_by:
        st.warning("Please select at least one column to group by.")
        return  # Stop execution if no grouping column is selected.

    df_agg = aggregate_data(df_filtered, selected_group_by)

    # 9. Display Aggregated Data
    st.subheader("Aggregated Data")
    st.dataframe(df_agg)

    # 10. Create Plot (with actual demand)
    st.subheader("Demand Forecast and Actual Demand Over Time")  # Added Actual Demand
    if 'Date' in df_agg.columns:
        fig = px.line(df_agg, x='Date', y=['demand_forecast', 'actual_demand'],  # Added actual_demand to the plot
                      title='Demand Forecast and Actual Demand Over Time',  # Modified title
                      labels={'value': 'Demand', 'Date': 'Date'},
                      line_group=selected_group_by if len(selected_group_by) > 1 else None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select 'date' as one of the grouping columns to visualize the trend over time.")

    # 11.  Show Performance Metrics
    st.subheader("Performance Metrics")
    # calculate and display the metrics.
    if not df_agg.empty:
        # Calculate total demand and total actual demand
        total_demand = df_agg['demand_forecast'].sum()
        total_actual_demand = df_agg['actual_demand'].sum()

        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(df_agg['demand_forecast'] - df_agg['actual_demand']))

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((df_agg['demand_forecast'] - df_agg['actual_demand']) ** 2))

        # Display the metrics
        st.write(f"Total Demand Forecast: {total_demand:.2f}")
        st.write(f"Total Actual Demand: {total_actual_demand:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    else:
        st.write("No data to display metrics.")

    # 12. Future Prediction Section
    st.header("Future Demand Prediction")
    st.write("Select parameters to predict future demand.")

    # 13. Prediction Inputs
    predict_product_id = st.selectbox("Product ID", df['Product ID'].unique())
    predict_store_id = st.selectbox("Store ID", df['Store ID'].unique())
    predict_category = st.selectbox("Category", df['Category'].unique())
    predict_region = st.selectbox("Region", df['Region'].unique())
    predict_date = st.date_input("Prediction Date", value=df['Date'].max() + pd.Timedelta(days=1))  # Default to next day

    # 14. Prediction Frequency
    prediction_frequency = st.radio(
        "Prediction Frequency",
        ('Next Day', 'Next Week', 'Next Month', 'Next Year')
    )

    # 15. Train a Model (Simplified for demonstration)
    # In a real application, you would use a more sophisticated model and feature engineering.
    df_model = df[['Date', 'Store ID', 'Product ID', 'Category', 'Region', 'Demand Forecast']]
    df_model['Date'] = pd.to_numeric(df_model['Date'])  # Convert date to numerical for the model
    X = df_model.drop('Demand Forecast', axis=1)
    y = df_model['Demand Forecast']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added a train test split
    model = LinearRegression()  # Using a simple linear regression model
    model.fit(X_train, y_train)

    # 16. Make Prediction
    def make_prediction(model, product_id, store_id, category, region, date):
        predict_df = pd.DataFrame({
            'Date': [pd.Timestamp(date).value],  # fixed here
            'Store ID': [store_id],
            'Product ID': [product_id],
            'Category': [category],
            'Region': [region],
        })
        prediction = model.predict(predict_df)
        return prediction[0]

        # Adjust prediction date
    if prediction_frequency == 'Next Day':
        adjusted_date = predict_date + pd.Timedelta(days=1)
    elif prediction_frequency == 'Next Week':
        adjusted_date = predict_date + pd.Timedelta(weeks=1)
    elif prediction_frequency == 'Next Month':
        adjusted_date = predict_date + pd.DateOffset(months=1)
    elif prediction_frequency == 'Next Year':
        adjusted_date = predict_date + pd.DateOffset(years=1)


     # Make prediction
    if st.button("Predict Demand"):
        predicted_demand = make_prediction(model, predict_product_id, predict_store_id, predict_category, predict_region, adjusted_date)
        st.subheader("Predicted Demand")

        if prediction_frequency=='Next Day':
            st.write(f"The predicted demand for the selected parameters for {prediction_frequency} is: {predicted_demand:.2f}")
        if prediction_frequency=='Next Week':
            st.write(f"The predicted demand for the selected parameters for {prediction_frequency} is: {predicted_demand*7:.2f}")
        if prediction_frequency=='Next Month':
            st.write(f"The predicted demand for the selected parameters for {prediction_frequency} is: {predicted_demand*30:.2f}")
        if prediction_frequency=='Next Year':
            st.write(f"The predicted demand for the selected parameters for {prediction_frequency} is: {predicted_demand*365:.2f}")

        #st.write(f"The predicted demand for the selected parameters for {prediction_frequency} is: {predicted_demand:.2f}")
        st.write(f"Product ID: {predict_product_id}, Store ID: {predict_store_id}, Category: {predict_category}, Region: {predict_region}, Date: {predict_date}") #Added to display the selected parameters

if __name__ == "__main__":
    main()
