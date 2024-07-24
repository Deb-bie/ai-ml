# Import the required libraries
import numpy as np # linear algebra
import pandas as pd # for data processing, CSV file I/O (e.g. pd.read_csv) - for reading csv files into the application


# Load the csv file as a data frame.
df = pd.read_csv('D://My files//ML//edureka//weatherforecast//weather.csv')
print('Size of weather data frame is :', df.shape)


# Display the data
print(df[0:5])

# Data Pre-processing
# Checking null values
# print(df.count().sort_values())


# As we can see, the first 2 columns have less than  98% data, we can ignore these 4 columns.
# We need to drop RISK_mm because we want to predict 'Rain Tomorrow' and RIST_MM can leak some information (the amount of rain that can occur the next day)
df = df.drop(columns=['RISK_MM', 'WindDir9am', 'WindSpeed9am'], axis=1)
# print(df)

# Let us get rid of all null values in df
df = df.dropna(how='any')
print(df.shape)


# its time to remove outliners in our data - we are using z-score to detect and remove the outliners
# an outlier is a data point that is very different from your other observations
from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df = df[(z < 3).all(axis=1)]
print(df.shape)


# Lets deal with the categorical columns now
# simply change yes/no to 1/0 for RainToday and RainTomorrow
df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)


#

#


#