import requests
import pandas as pd
from pandas import json_normalize
import numpy as np
import datetime
from fbprophet.plot import plot_plotly, plot_components_plotly
from fbprophet import Prophet
pd.options.display.float_format = "{:,.0f}".format

retrieveDataFromDate = datetime.date(2019, 10, 18)
baseUrl = "https://data.gov.gr/api/v1/query/internet_traffic?date_from="
addUrl = "&date_to="

df = pd.DataFrame()
while True:
    print("Retrieving data for " + str(retrieveDataFromDate))
    url = (
        baseUrl
        + str(retrieveDataFromDate)
        + addUrl
        + str(retrieveDataFromDate + datetime.timedelta(days=30))
    )
    headers = {"Authorization": "Token f5f91f83b6789f0aa9b9572e64696cfde984b7eb"}
    response = requests.get(url, headers=headers)
    monthlydf = pd.json_normalize(response.json())
    try:
        if len(monthlydf) <= 5:
            break
    except:
        break
    retrieveDataFromDate = retrieveDataFromDate + datetime.timedelta(days=31)
    df = pd.concat([df, monthlydf])

df = df[["date", "avg_in"]].rename(columns={"avg_in": "traffic"})

df = df.sort_values(by="date")

df["date"] = pd.to_datetime(df["date"]).dt.date
df = df.groupby("date").mean().reset_index()

df.to_pickle('dataset.pkl')

df=pd.read_pickle('dataset.pkl')
df=df.rename(columns={'date':'ds','traffic':'y'})
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


