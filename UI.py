import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ==================== CONFIG ====================
st.set_page_config(layout="wide")

# ==================== DATA ====================
@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    return prepare_data(df)


def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'])

    df['Store ID'] = df['Store ID'].map({'S001':1,'S002':2,'S003':3,'S004':4,'S005':5})
    df['Product ID'] = df['Product ID'].str.replace('P','').astype(int)
    df['Category'] = df['Category'].map({'Groceries':1,'Toys':2,'Electronics':3,'Furniture':4,'Clothing':5})
    df['Region'] = df['Region'].map({'North':1,'South':2,'West':3,'East':4})

    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Weekday'] = df['Date'].dt.day_name()
    return df

# ==================== APP ====================
def main():
    st.title("Demand Forecasting Dashboard")

    df = load_data()

    # -------- SIDEBAR FILTERS --------
    with st.sidebar:
        st.header("Filters")
        product = st.multiselect("Product", df['Product ID'].unique(), df['Product ID'].unique())
        store = st.multiselect("Store", df['Store ID'].unique(), df['Store ID'].unique())
        category = st.multiselect("Category", df['Category'].unique(), df['Category'].unique())
        region = st.multiselect("Region", df['Region'].unique(), df['Region'].unique())
        date_range = st.date_input("Date Range", [df['Date'].min(), df['Date'].max()])

    df_f = df[
        (df['Product ID'].isin(product)) &
        (df['Store ID'].isin(store)) &
        (df['Category'].isin(category)) &
        (df['Region'].isin(region)) &
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1])
    ]

    # -------- GROUPING --------
    group_cols = st.multiselect(
        "Group By",
        ['Date','Product ID','Store ID','Category','Region','Month','Year','Weekday'],
        default=['Date']
    )

    agg = df_f.groupby(group_cols).agg(
        demand_forecast=('Demand Forecast','sum'),
        actual_demand=('Units Sold','sum')
    ).reset_index()

    # -------- KPIs --------
    total_f = agg['demand_forecast'].sum()
    total_a = agg['actual_demand'].sum()
    mae = np.mean(np.abs(agg['demand_forecast'] - agg['actual_demand']))
    rmse = np.sqrt(np.mean((agg['demand_forecast'] - agg['actual_demand'])**2))

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Forecast", f"{total_f:.0f}")
    k2.metric("Total Actual", f"{total_a:.0f}")
    k3.metric("MAE", f"{mae:.2f}")
    k4.metric("RMSE", f"{rmse:.2f}")

    # -------- DATA + CHART --------
    left, right = st.columns([1.2,2])

    with left:
        st.subheader("Aggregated Data")
        st.dataframe(agg, height=350, use_container_width=True)

    with right:
        st.subheader("Forecast vs Actual")
        if 'Date' in agg.columns:
            fig = px.line(
                agg,
                x='Date',
                y=['demand_forecast','actual_demand'],
                labels={'value':'Demand'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select Date to see time series")

    # -------- TABS --------
    tab1, tab2, tab3 = st.tabs(["Prediction","Group Info","Mappings"])

    # -------- TAB 1 : PREDICTION --------
    with tab1:
        model_df = df[['Date','Store ID','Product ID','Category','Region','Demand Forecast']].copy()
        model_df['Date'] = model_df['Date'].astype('int64')

        X = model_df.drop('Demand Forecast',axis=1)
        y = model_df['Demand Forecast']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        model = LinearRegression()
        model.fit(X_train,y_train)

        c1,c2 = st.columns(2)
        with c1:
            p_prod = st.selectbox("Product", df['Product ID'].unique())
            p_store = st.selectbox("Store", df['Store ID'].unique())
            p_date = st.date_input("Date")
        with c2:
            p_cat = st.selectbox("Category", df['Category'].unique())
            p_reg = st.selectbox("Region", df['Region'].unique())
            freq = st.radio("Frequency", ['Next Day','Next Week','Next Month','Next Year'], horizontal=True)

        if st.button("Predict Demand"):
            base_date = pd.Timestamp(p_date)
            delta = {
                'Next Day': pd.Timedelta(days=1),
                'Next Week': pd.Timedelta(weeks=1),
                'Next Month': pd.DateOffset(months=1),
                'Next Year': pd.DateOffset(years=1)
            }[freq]

            d = base_date + delta

            Xp = pd.DataFrame({
                'Date':[d.value],
                'Store ID':[p_store],
                'Product ID':[p_prod],
                'Category':[p_cat],
                'Region':[p_reg]
            })

            pred = model.predict(Xp)[0]
            multiplier = {'Next Day':1,'Next Week':7,'Next Month':30,'Next Year':365}[freq]
            st.metric("Predicted Demand", f"{pred*multiplier:.2f}")

    # -------- TAB 2 : GROUP INFO --------
    with tab2:
        st.markdown("**Current Grouping Columns**")
        st.write(group_cols)
        st.markdown("Grouping controls aggregation granularity. Multiple selections increase dimensionality.")

    # -------- TAB 3 : MAPPINGS --------
    with tab3:
        mapping_df = pd.DataFrame({
            'Feature': ['Store ID','Category','Region'],
            'Mapping': ['S001–S005 → 1–5','Groceries–Clothing → 1–5','North–East → 1–4']
        })
        st.dataframe(mapping_df, height=300, use_container_width=True)


if __name__ == '__main__':
    main()