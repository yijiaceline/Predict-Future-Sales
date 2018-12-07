import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math


#load data

train = pd.read_csv("../data/sales_train.csv")
test = pd.read_csv("../data/test.csv")

#data preprocessing
print(train.shape)
#missing value
train.isnull().sum()

#duplicate value
train[train.duplicated(keep='first')].count()

#Outliers
plt.figure(figsize=(10,4))
plt.xlim(-200, 3000)
sns.boxplot(x = train.item_cnt_day)

train = train.loc[(train.item_cnt_day < 2000)]
plt.show()

#transform the dataset in order to put in the model
train.date = train.date.apply(lambda x:dt.datetime.strptime(x, '%d.%m.%Y'))
train.date = train.date.apply(lambda x:dt.datetime.strftime(x,'%Y-%m'))
data = train.groupby(['date','item_id','shop_id']).sum().reset_index()

data = data[['date','item_id','shop_id','item_cnt_day']]
table = pd.pivot_table(data, values='item_cnt_day', index=['item_id', 'shop_id'],
                        columns=['date'], aggfunc=np.sum).reset_index()
table = table.fillna(0)

#get rid of those not in test data
data_inc = test.merge(table, on = ['item_id', 'shop_id'], how = 'left')
data_inc = data_inc.fillna(0)
data_inc = data_inc.iloc[:,3:]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

look_back = 5 #change lookback here to test
def create_dataset(data, look_back):
    dataX, dataY = [], []
    for i in range(len(data)-look_back):
        dataX.append(data[i:(i+look_back)])
        dataY.append(data[i + look_back])
    return np.array(dataX), np.array(dataY)

# data_inc.shape[0]

# normalize the dataset
norm = scaler.fit_transform(data_inc.values[374].reshape(-1,1)) #choose a random product
dataX, dataY = create_dataset(norm, look_back)

split = data_inc.shape[1] - look_back -1
X_train = dataX[:split]
X_test = dataX[split:]
Y_train = dataY[:split]
Y_test = dataY[split:]

#reshape to [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

def model_1(): #add lookback
    model = Sequential()
    model.add(LSTM(128, input_shape=(1,look_back)))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    adam = optimizers.Adam()
    model.compile(loss='mse', optimizer=adam, metrics=['mean_squared_error'])

    # model.summary()

    return model

model1 = model_1()

#record training time
start=time.time()
training1=model1.fit(X_train, Y_train, batch_size = 1, epochs = 20, shuffle=False)
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)

# make predictions
trainPredict = model1.predict(X_train)
testPredict = model1.predict(X_test)

#test error
test_rmse = model1.evaluate(X_test, Y_test)
print("--------------------------")
print('RMSE' , test_rmse[0])

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(Y_train)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(Y_test)

# calculate root mean squared error
plt.plot(training1.history['loss'], label= 'loss(mse)')
plt.plot(np.sqrt(training1.history['mean_squared_error']), label= 'rmse')
plt.title('Training loss')
plt.legend(loc=1)

plt.plot(data_inc.values[374])
plt.xlabel('month')
plt.ylabel('sales')
plt.title('sales record for item 15238 in shop 5')

plt.show()
