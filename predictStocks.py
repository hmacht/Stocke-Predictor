import quandl
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import math
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation
from sklearn.cross_validation import train_test_split
import pickle
quandl.ApiConfig.api_key = "6zHM9EPfjZD6qk76sYBV"
style.use('ggplot')
df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PTC_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PTC_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# predicting 10% of data out
forecast_out = int(math.ceil(0.01*len(df)))
#df['Adj. Close'].plot()
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
print(X_lately)
df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



#Train

##regressor = LinearRegression()
##regressor.fit(X_train, y_train)
##with open('linearregression.pickle', 'wb') as f:
##    pickle.dump(regressor, f)

pickle_in = open('linearregression.pickle', 'rb')
regressor = pickle.load(pickle_in)

accuracy = regressor.score(X_test, y_test)
forecase_set = regressor.predict(X_lately)
print(forecase_set, accuracy, forecast_out)
df['Forecast'] = np.nan
#print(df.tail())
print(df)


# Get date X and y data dosent corrrespond to axis
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecase_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


# Graph
df['Adj. Close'].plot()
df['Forecast'].plot()
#df['label'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
