import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation,performance_metrics
from prophet.plot import plot_cross_validation_metric,plot_plotly, plot_components_plotly
from pycaret.time_series import *
import requests
import datetime
import altair as alt
pd.options.display.float_format = "{:,.3f}".format



#########################################################################################
#Download Dataset from data.gov.gr
#due to days limitation, the code download each time 30 days data.
retrieveDataFromDate = datetime.date(2019, 10, 18)
baseUrl = "https://data.gov.gr/api/v1/query/internet_traffic?date_from="
addUrl = "&date_to="


df = pd.DataFrame()
while True:
    print("Retrieving data for "
         + str(retrieveDataFromDate)
         +" up to "
         + str(retrieveDataFromDate 
            + datetime.timedelta(days=30)))
    url = (
        baseUrl
        + str(retrieveDataFromDate)
        + addUrl
        + str(retrieveDataFromDate + datetime.timedelta(days=30))
    )
    headers = {"Authorization": "Token --"}
    response = requests.get(url, headers=headers)
    monthlydf = pd.json_normalize(response.json())
    try:
        if len(monthlydf) <= 2:
            break
    except:
        break
    retrieveDataFromDate = retrieveDataFromDate + datetime.timedelta(days=31)
    df = pd.concat([df, monthlydf])
df.to_pickle('pre_dataset.pkl')


#########################################################################################
#Basic Preprocessing
#Applying 7 Days Rolling
df=pd.read_pickle('pre_dataset.pkl')

df = df[["date", "avg_in"]].rename(columns={"avg_in": "traffic"})
df = df.sort_values(by="date")

df["date"] = pd.to_datetime(df["date"]).dt.date
df = df.groupby("date").mean().reset_index()
alt.Chart(df[['traffic']].reset_index()).mark_line().encode(
    x='index',
    y='traffic'
)
df['traffic_normalized']=df['traffic'].rolling(window=7, min_periods=1).mean()
df=df[['date','traffic_normalized']]
alt.Chart(df[['traffic_normalized']].reset_index()).mark_line().encode(
    x='index',
    y='traffic_normalized'
)
#exporting dataset as pickle
df.to_pickle('dataset.pkl')
df=pd.read_pickle('dataset.pkl')

#########################################################################################
#Apply Facebook Prophet Forecast
df=df.rename(columns={'date':'ds','traffic_normalized':'y'})
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

#########################################################################################
# Results Visualization
plot_components_plotly(m, forecast)
plot_plotly(m, forecast)
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '100 days')
cutoffs = pd.to_datetime(['2020-01-01', '2021-01-01', '2021-10-01'])
df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='100 days')
df_p = performance_metrics(df_cv)
plot_cross_validation_metric(df_cv, metric='mape')


#########################################################################################
#PyCaret Time Series
data=pd.read_pickle('dataset.pkl')
data=data[600:]
data['date']=pd.to_datetime(data['date'])
data.set_index('date',inplace=True,drop=True)
data=data.asfreq('D')
s = setup(data, fh=28,fold=3,session_id=1
)
best = compare_models(sort='mae')
#########################################################################################
#PyCaret Results Visualization
plot_model(best, plot = 'forecast', data_kwargs = {'fh' : 365})
plot_model(best, plot = 'diagnostics')
plot_model(best, plot = 'insample')
predict_model(best, fh = 365)

#########################################################################################
#PyCaret Time Series by month and not daily
data=pd.read_pickle('dataset.pkl')

data['date']=pd.to_datetime(data['date'])
data.set_index('date',inplace=True,drop=True)
data=data.asfreq('D')
data=data.groupby(pd.Grouper(freq='M')).mean()
s = setup(data, fh=7,session_id=1
)
best = compare_models(sort='mape')
#########################################################################################
#PyCaret Results Visualization
plot_model(best, plot = 'forecast', data_kwargs = {'fh' : 24})
plot_model(best, plot = 'diagnostics')
plot_model(best, plot = 'insample')
predict_model(best, fh = 7)



