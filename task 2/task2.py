Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
data=pd.read_csv("C:\\Users\\madak\\OneDrive\\Desktop\\Unemployment_Rate_upto_11_2020.csv")
print(data.head())
           Region         Date  Frequency  ...  Region.1  longitude  latitude
0  Andhra Pradesh   31-01-2020          M  ...     South    15.9129     79.74
1  Andhra Pradesh   29-02-2020          M  ...     South    15.9129     79.74
2  Andhra Pradesh   31-03-2020          M  ...     South    15.9129     79.74
3  Andhra Pradesh   30-04-2020          M  ...     South    15.9129     79.74
4  Andhra Pradesh   31-05-2020          M  ...     South    15.9129     79.74

[5 rows x 9 columns]
#check missing values
print(data.isnull().sum())
Region                                      0
 Date                                       0
 Frequency                                  0
 Estimated Unemployment Rate (%)            0
 Estimated Employed                         0
 Estimated Labour Participation Rate (%)    0
Region.1                                    0
longitude                                   0
latitude                                    0
dtype: int64
>>> #correlation between features of dataset
>>> data.columns= ["States","Date","Frequency","Estimated Unemployment Rate","Estimated Employed","Estimated Labour Participation Rate","Region","longitude","latitude"]
>>> sns.set(style="whitegrid")
>>> plt.figure(figsize=(12, 10))
<Figure size 1200x1000 with 0 Axes>
>>> sns.heatmap(data.corr())

Warning (from warnings module):
  File "<pyshell#13>", line 1
FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
<AxesSubplot: >
>>> plt.show()
>>> 
>>> #unemployment rate analysis
>>> #estimated number of employees by region
>>> data.columns= ["States","Date","Frequency","Estimated Unemployment Rate","Estimated Employed","Estimated Labour Participation Rate","Region","longitude","latitude"]
>>> plt.title("Indian Unemployment")
Text(0.5, 1.0, 'Indian Unemployment')
>>> sns.histplot(x="Estimated Employed", hue="Region", data=data)
<AxesSubplot: title={'center': 'Indian Unemployment'}, xlabel='Estimated Employed', ylabel='Count'>
>>> plt.show()
>>> 
>>> #unemployment rate by region
>>> plt.figure(figsize=(12, 10))
<Figure size 1200x1000 with 0 Axes>
>>> plt.title("Indian Unemployment")
Text(0.5, 1.0, 'Indian Unemployment')
>>> sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=data)
<AxesSubplot: title={'center': 'Indian Unemployment'}, xlabel='Estimated Unemployment Rate', ylabel='Count'>
>>> plt.show()
>>> 
>>> #unemployment rate by state
>>> unemploment = data[["States", "Region", "Estimated Unemployment Rate"]]
>>> figure = px.sunburst(unemploment, path=["Region", "States"],values="Estimated Unemployment Rate",width=700, height=700, color_continuous_scale="RdY1Gn",title="Unemployment Rate in India")
>>> figure.show()
