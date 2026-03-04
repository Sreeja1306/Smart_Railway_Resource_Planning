import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

# --- Configuration & Setup ---
st.set_page_config(page_title="SmartRail Dashboard", layout="wide", page_icon="🚆")

st.title("SmartRail: Passenger Demand & Resource Allocation")
st.markdown("A practical planning tool built to help railway operations teams make smarter decisions about trains, coaches, and platforms.")

# --- Data Loading ---
@st.cache_data
def load_data():
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv", parse_dates=["date"])
        df['occupancy_rate'] = df['passenger_count'] / df['seat_capacity']
        return df

    # Generate dataset only if CSV does not exist
    np.random.seed(42)
    n = 300
    routes    = ["Mumbai-Delhi", "Chennai-Bangalore", "Kolkata-Patna",
                 "Delhi-Jaipur", "Hyderabad-Pune", "Ahmedabad-Surat"]
    day_types = ["Weekday", "Weekend", "Holiday"]

    df = pd.DataFrame({
        "train_id"        : [f"T{str(i).zfill(3)}" for i in range(1, n+1)],
        "route"           : np.random.choice(routes, n),
        "date"            : pd.date_range("2024-01-01", periods=n, freq="D"),
        "day_type"        : np.random.choice(day_types, n),
        "departure_time"  : [f"{h:02d}:00" for h in np.random.randint(5, 23, n)],
        "passenger_count" : np.random.randint(100, 500, n),
        "seat_capacity"   : np.random.randint(300, 600, n),
        "num_coaches"     : np.random.randint(6, 20, n),
        "platform_number" : np.random.randint(1, 10, n),
        "delay_minutes"   : np.random.randint(0, 60, n),
    })
    df['date']           = pd.to_datetime(df['date'])
    df['occupancy_rate'] = df['passenger_count'] / df['seat_capacity']

    # Save to CSV so it is reused on every future run
    df.to_csv("dataset.csv", index=False)
    return df

df = load_data()

with st.expander("About the Dataset"):
    st.markdown(f"""
    - **Source:** Synthetically generated data simulating Indian railway operations
    - **Total Records:** {len(df):,} train entries
    - **Routes Covered:** {df['route'].nunique()} unique routes
    - **Date Range:** {df['date'].min().date()} to {df['date'].max().date()}
    - **Fields:** Train ID, Route, Date, Day Type, Passenger Count, Seat Capacity, Coaches, Platform, Delay
    - **Note:** Dataset is generated once on first run and saved as dataset.csv for consistency
    """)

# ==========================================
# Section 1: Overview Dashboard (KPIs)
# ==========================================
st.markdown("---")
st.header("1. Overview Dashboard")
st.markdown("A quick snapshot of the current data — total trains, passenger load, and peak demand points.")

col1, col2, col3, col4 = st.columns(4)

total_trains = len(df)
total_passengers = df['passenger_count'].sum()
avg_occupancy = df['occupancy_rate'].mean()

busiest_route = df.groupby('route')['passenger_count'].sum().idxmax()
busiest_time = df.groupby('departure_time')['passenger_count'].sum().idxmax()

col1.metric("Total Trains", f"{total_trains:,}")
col2.metric("Total Passengers", f"{total_passengers:,}")
col3.metric("Avg Occupancy Rate", f"{avg_occupancy * 100:.1f}%")
col4.metric("Busiest Route", busiest_route, f"Peak Time: {busiest_time}")


# ==========================================
# Section 2: Passenger Demand Visualization
# ==========================================
st.markdown("---")
st.header("2. Passenger Demand Insights")
st.markdown("Explore which routes carry the most passengers, how demand shifts over time, and whether weekends or holidays drive peak loads.")

viz_col1, viz_col2, viz_col3 = st.columns(3)

with viz_col1:
    # Bar Chart
    df_route = df.groupby('route')['passenger_count'].sum().reset_index()
    fig1 = px.bar(df_route, x='route', y='passenger_count', title="Which routes carry the most passengers?", color='route')
    st.plotly_chart(fig1, use_container_width=True)

with viz_col2:
    # Line Chart
    df_time = df.groupby('date')['passenger_count'].sum().reset_index()
    fig2 = px.line(df_time, x='date', y='passenger_count', title="How has daily demand changed over time?")
    st.plotly_chart(fig2, use_container_width=True)

with viz_col3:
    # Day Type Heatmap / Pie
    df_day = df.groupby('day_type')['passenger_count'].mean().reset_index()
    fig3 = px.pie(df_day, names='day_type', values='passenger_count', title="Do weekends and holidays drive more passengers?", hole=0.4)
    st.plotly_chart(fig3, use_container_width=True)


# ==========================================
# Section 3: Machine Learning Prediction
# ==========================================
st.markdown("---")
st.header("3. Demand Prediction Engine")
st.markdown("Select a route and day type to see what demand might look like — useful for planning ahead of peak seasons.")

# Prepare historical data to train the prediction model
# Simple model: predict passenger count based on route and day_type
ml_df = df[['route', 'day_type', 'passenger_count']].dropna()
X = ml_df[['route', 'day_type']]
y = ml_df['passenger_count']

# One-hot encode categorical features route & day_type
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# Train a simple linear regression on route and day type
model = LinearRegression()
model.fit(X_encoded, y)

pred_col1, pred_col2 = st.columns([1, 2])

with pred_col1:
    st.subheader("Select Parameters")
    
    # Input selections
    routes = sorted(df['route'].unique())
    day_types = sorted(df['day_type'].unique())
    
    sel_route = st.selectbox("Select Route:", routes)
    sel_day = st.selectbox("Select Day Type:", day_types)
    
    if st.button("Predict Passenger Demand", type="primary"):
        # Run prediction using selected route and day type
        input_data = pd.DataFrame({'route': [sel_route], 'day_type': [sel_day]})
        input_encoded = encoder.transform(input_data)
        prediction = model.predict(input_encoded)[0]
        
        # Determine actual historical average to compare
        hist_avg = df[(df['route'] == sel_route) & (df['day_type'] == sel_day)]['passenger_count'].mean()
        
        st.success(f"**Predicted Passengers:** {int(prediction):,}")
        
        with pred_col2:
            st.subheader("Predicted vs Historical Average")
            comp_df = pd.DataFrame({
                'Metric': ['Predicted Demand', 'Historical Average'],
                'Passengers': [int(prediction), int(hist_avg) if pd.notna(hist_avg) else 0]
            })
            fig_pred = px.bar(comp_df, x='Metric', y='Passengers', color='Metric', text='Passengers')
            st.plotly_chart(fig_pred, use_container_width=True)


# ==========================================
# Section 4: Resource Recommendations
# ==========================================
st.markdown("---")
st.header("4. Resource Recommendations")
st.markdown("Each train is reviewed against occupancy and delay thresholds to flag where action is needed.")
st.markdown("""
Trains running above 85% capacity are flagged for extra coaches.
Trains below 40% capacity are marked for reduction.
Any train delayed over 15 minutes is flagged for platform review.
""")

# Flag each train based on occupancy and delay thresholds
rec_df = df.copy()

def get_coach_rec(occ):
    if occ > 0.85: return "Add Coaches ⬆️"
    elif occ < 0.40: return "Reduce Coaches ⬇️"
    return "Optimal 🟢"

def get_plat_rec(delay):
    if delay > 15: return "Reassign Platform ⚠️"
    return "Stable 🟢"

rec_df['Coach_Recommendation'] = rec_df['occupancy_rate'].apply(get_coach_rec)
rec_df['Platform_Recommendation'] = rec_df['delay_minutes'].apply(get_plat_rec)

# Select only the columns relevant for the recommendations table
display_df = rec_df[['train_id', 'route', 'date', 'occupancy_rate', 'delay_minutes', 'Coach_Recommendation', 'Platform_Recommendation']].copy()

# Format for display
display_df['occupancy_rate'] = (display_df['occupancy_rate'] * 100).round(1).astype(str) + '%'
display_df = display_df.head(50) # Show top 50 rows for performance

# Style functions
def style_recommendations(row):
    color_map = []
    for val in row:
        if isinstance(val, str) and "Add Coaches ⬆️" in val: color_map.append('background-color: #ffcccc; color: #990000')
        elif isinstance(val, str) and "Reduce Coaches ⬇️" in val: color_map.append('background-color: #ffe5cc; color: #cc6600')
        elif isinstance(val, str) and "Reassign Platform ⚠️" in val: color_map.append('background-color: #fce4ec; color: #c2185b')
        elif isinstance(val, str) and ("Optimal" in val or "Stable" in val): color_map.append('background-color: #ccffcc; color: #006600')
        else: color_map.append('')
    return color_map

st.dataframe(
    display_df.style.apply(style_recommendations, axis=1),
    use_container_width=True,
    hide_index=True
)
