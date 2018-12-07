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


#load data
train = pd.read_csv("../data/sales_train.csv")
test = pd.read_csv("../data/test.csv")

train = train.loc[(train.item_cnt_day < 2000)]

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
dataset = scaler.fit_transform(data_inc.values.T)
dataset = dataset.T

X, y = dataset[:,:33], dataset[:,33:]

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
#check shape of x_train and y_train
# reshape input to be [samples, time steps, features]
x_train = x_train.reshape((149940, 33, 1))
y_train = y_train.reshape((149940,1))
x_test = x_test.reshape((64260, 33, 1))
y_test = y_test.reshape((64260,1))




batch_size = 1260
look_back = 33
def model_1(): #change neurons
    model = Sequential()
    model.add(LSTM(64, batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),
                   stateful = False))
    model.add(Dense(1))
    adam = optimizers.Adam()
    model.compile(loss='mse', optimizer=adam, metrics=['mean_squared_error'])

    return model

model1 = model_1()

#record training time
start=time.time()
training1=model1.fit(x_train, y_train, batch_size = 1260, epochs = 20, shuffle=False)
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)

# make predictions
trainPredict = model1.predict(x_train, batch_size = 1260)
testPredict = model1.predict(x_test,batch_size = 1260)

#test error
test_rmse = model1.evaluate(x_test, y_test,batch_size = 1260)
print("--------------------------")
print('RMSE' , test_rmse[0])






batch_size = 1260
look_back = 33
def model_4(): #change neurons
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),
                   stateful = False))
    model.add(Dense(1))
    adam = optimizers.Adam()
    model.compile(loss='mse', optimizer=adam, metrics=['mean_squared_error'])

    return model

model4 = model_4()

#record training time
start=time.time()
training4=model4.fit(x_train, y_train, batch_size = 1260, epochs = 20, shuffle=False)
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)

# make predictions
trainPredict = model4.predict(x_train, batch_size = 1260)
testPredict = model4.predict(x_test,batch_size = 1260)

#test error
test_rmse = model4.evaluate(x_test, y_test,batch_size = 1260)
print("--------------------------")
print('RMSE' , test_rmse[0])




plt.plot(training4.history['loss'], label= 'loss(mse)')
plt.plot(np.sqrt(training4.history['mean_squared_error']), label= 'rmse')
plt.title('Training loss')
plt.ylim(0,0.3)
plt.legend(loc=1)
plt.show()