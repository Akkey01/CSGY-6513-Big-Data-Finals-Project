import streamlit as st
st.set_page_config(page_title="🚇 MTA Ridership Explorer", layout='wide')
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
import folium
from prophet import Prophet
import json

# ---
# Dashboard: MTA Ridership Explorer
# Dependencies:
#   pip install streamlit pandas plotly streamlit-folium folium prophet
# ---

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['transit_timestamp'])
    df['hour'] = df['transit_timestamp'].dt.hour
    return df

@st.cache_data
def prep_forecast(df_station: pd.DataFrame, days: int = 7):
    ts = (
        df_station
        .set_index('transit_timestamp')['ridership']
        .resample('D').sum()
        .reset_index()
        .rename(columns={'transit_timestamp': 'ds', 'ridership': 'y'})
    )
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(ts)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return ts, forecast

# Load and filter data
df_master = load_data("master_df.csv")
min_date = df_master['transit_timestamp'].dt.date.min()
max_date = df_master['transit_timestamp'].dt.date.max()
date_range = st.sidebar.date_input(
    "Date range", [min_date, max_date],
    min_value=min_date, max_value=max_date
)
filtered = df_master.loc[
    (df_master['transit_timestamp'].dt.date >= date_range[0]) &
    (df_master['transit_timestamp'].dt.date <= date_range[1])
]
stations = filtered['station_complex'].unique().tolist()
selected_station = st.sidebar.selectbox("Select station", stations)
df_station = filtered[filtered['station_complex'] == selected_station]

# App title
st.title(f"🚇 MTA Ridership Explorer: {selected_station}")

# 1. Hourly Ridership
st.header("Ridership Over Time")
hourly = df_station.groupby('hour')['ridership'].sum().reset_index()
fig_h = px.bar(hourly, x='hour', y='ridership', title='Hourly Ridership',
               color='hour', color_continuous_scale='Viridis')
fig_h.update_layout(xaxis_title='Hour', yaxis_title='Ridership')
st.plotly_chart(fig_h, use_container_width=True)

# 2. Daily & Weekly Ridership

daily = df_station.set_index('transit_timestamp')['ridership'].resample('D').sum().reset_index()
weekly = df_station.set_index('transit_timestamp')['ridership'].resample('W').sum().reset_index()

fig_daily = px.line(daily, x='transit_timestamp', y='ridership', title='Daily Ridership',
                    color_discrete_sequence=['#636EFA'])
fig_daily.update_layout(xaxis_title='Date', yaxis_title='Ridership')
st.plotly_chart(fig_daily, use_container_width=True)

fig_weekly = px.line(weekly, x='transit_timestamp', y='ridership', title='Weekly Ridership',
                     color_discrete_sequence=['#EF553B'])
fig_weekly.update_layout(xaxis_title='Date', yaxis_title='Ridership')
st.plotly_chart(fig_weekly, use_container_width=True)

# 3. Station Location
st.header("Station Location")
lat, lon = df_station['latitude'].mean(), df_station['longitude'].mean()
m = folium.Map(location=(lat, lon), zoom_start=14)
folium.CircleMarker((lat, lon), radius=10, popup=selected_station,
                    color='darkblue', fill=True, fill_color='skyblue').add_to(m)
st_folium(m, width=700)

# 4. 7-day Forecast
st.header("7-day Forecast")
with st.spinner("Training forecasting model..."):
    ts, forecast = prep_forecast(df_station)
fig_fc = px.line(forecast, x='ds', y='yhat', title='7-day Forecast',
                 color_discrete_sequence=['#00CC96'])
fig_fc.add_scatter(x=ts['ds'], y=ts['y'], mode='markers', name='Actual', marker_color='black')
fig_fc.update_layout(xaxis_title='Date', yaxis_title='Ridership')
st.plotly_chart(fig_fc, use_container_width=True)

# 5. Borough-level Summary
st.header("Borough-level Summary")
boroughs = filtered['borough'].unique().tolist()
tabs = st.tabs(boroughs)
for tab, b in zip(tabs, boroughs):
    with tab:
        df_b = filtered[filtered['borough'] == b]
        st.markdown(f"**{b}** ({df_b['station_complex'].nunique()} stations)")
        stations_b = df_b['station_complex'].unique().tolist()
        sel = st.multiselect(f"Stations in {b}", stations_b, default=stations_b[:3])
        if sel:
            df_sel = df_b[df_b['station_complex'].isin(sel)]
            df_plot = (
                df_sel.set_index('transit_timestamp')
                      .groupby('station_complex')['ridership']
                      .resample('D').sum().reset_index()
            )
            fig = px.line(df_plot, x='transit_timestamp', y='ridership',
                          color='station_complex', title=f'Daily Trends in {b}',
                          color_discrete_sequence=px.colors.qualitative.T10)
            fig.update_layout(xaxis_title='Date', yaxis_title='Ridership')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one station to view trends.")

# 6. Payment & Fare Category Breakdown
if 'payment_method' in df_station.columns and 'fare_class_category' in df_station.columns:
    rev_df = (
        df_station.groupby(['payment_method', 'fare_class_category'])['ridership']
                  .sum().reset_index(name='revenue')
    )
    fig_sf = px.sunburst(
        rev_df,
        path=['payment_method', 'fare_class_category'],
        values='revenue',
        color='payment_method',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title='Revenue by Payment Method & Fare Category'
    )
    fig_sf.update_traces(textinfo='label+percent entry', insidetextorientation='radial')
    fig_sf.update_layout(width=800, height=600, margin=dict(t=50, l=0, r=0, b=0))
    st.plotly_chart(fig_sf, use_container_width=False)

# 7. Ridership vs Distance to Central
if 'distance_to_central' in df_station.columns:
    samp = df_station.sample(min(len(df_station), 2000))
    fig_sc = px.scatter(samp, x='distance_to_central', y='ridership',
                        color='borough', title='Ridership vs Distance to Central',
                        color_discrete_sequence=px.colors.qualitative.Dark24)
    fig_sc.update_layout(xaxis_title='Distance (m)', yaxis_title='Ridership')
    st.plotly_chart(fig_sc, use_container_width=True)

# Note: order of graphs preserved: Hourly, Daily, Weekly, Map, Forecast, Borough, Payment, Distance
