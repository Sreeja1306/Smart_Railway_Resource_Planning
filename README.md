# SmartRail - Railway Resource Allocation System 🚆

A web app built using Streamlit that helps railway planners understand
passenger demand and decide where to add or reduce coaches and flag
platform issues. Built as part of a hackathon project.

## About the Dataset

The dataset is synthetically generated on the first run of the app
and saved as dataset.csv automatically. It has 300 rows simulating
Indian railway train data across 6 routes. Fields include passenger
count, seat capacity, number of coaches, platform number, and delay
minutes. The same dataset is used every time you run the app so
results stay consistent.

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn

## What the App Does

1. Overview section shows total trains, passengers, average occupancy
   and the busiest route at a glance.

2. Demand Insights shows bar chart by route, a daily trend line chart,
   and a pie chart comparing weekday vs weekend vs holiday demand.

3. Prediction section uses a simple Linear Regression model trained on
   route and day type to predict how many passengers are expected.
   You pick a route and day type and it gives you a number.

4. Recommendations table checks each train and flags it based on these
   rules:
   - Above 85% occupancy means add more coaches
   - Below 40% occupancy means reduce coaches
   - Delay above 15 minutes means reassign platform

## Assumptions

- Data is synthetic and randomly generated, not from a live source
- Coach recommendations use fixed thresholds, not ML
- The prediction model trains fresh every time the app starts

## How to Run

Clone the repo

    git clone https://github.com/Sreeja1306/Smart_Railway_Resource_Planning.git
    cd Smart_Railway_Resource_Planning

Install requirements

    pip install -r requirements.txt

Run the app

    streamlit run app.py

## Screenshots

![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)

Deployed on Streamlit Cloud.
