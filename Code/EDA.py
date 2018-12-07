import os 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pylab as plt
import matplotlib.dates as mdates
plt.rcParams['figure.figsize'] = (15.0, 8.0)
import seaborn as sns
import datetime as dt

#load data
train = pd.read_csv("data/sales_train.csv")
test = pd.read_csv("data/test.csv")
cat = pd.read_csv("data/item_categories.csv")
items = pd.read_csv("data/items.csv")
shops = pd.read_csv("data/shops.csv")


#Feature Engineering
print('Number of unique in train dataset')
print(train.nunique())
print('-------------------------------------')
print('Number of unique in test dataset')
print(test.nunique())

print ('number of shops: ', train['shop_id'].max())
print ('number of items: ', train['item_id'].max())
num_month = train['date_block_num'].max()
print ('number of month: ', num_month)
print ('size of training dataset: ', train.shape)
print ('number of categories: ', items['item_category_id'].max()) # the maximun number of category id



data = pd.merge(train, items[['item_id','item_category_id']], on = 'item_id', how = 'left')
data.date=data.date.apply(lambda x:dt.datetime.strptime(x, '%d.%m.%Y'))
data['month'] = data['date'].dt.month



#figure 1
sales = data.groupby(["date_block_num"],as_index = False)["item_cnt_day"].sum()

plt.figure(figsize=(16,8))
plt.title('Total Sales of the company')
sns.lineplot(x = 'date_block_num', y = 'item_cnt_day', data = sales)
plt.xlabel('month')
plt.ylabel('Sales')

#figure 2
x = data.groupby(["shop_id"],as_index = False)["item_cnt_day"].sum()
x = x.sort_values(by='item_cnt_day',ascending=False)
x = x.iloc[0:10]
top10shop = x.shop_id.tolist()

temp = pd.DataFrame()
for i in top10shop:
    x = data.loc[(data.shop_id == i)]
    temp = temp.append(x)

temp = temp.groupby(["date_block_num","shop_id"],as_index = False)["item_cnt_day"].sum()    
    
plt.figure(figsize=(16,8))
sns.lineplot(x = 'date_block_num', y = 'item_cnt_day',data = temp, hue = 'shop_id', legend = 'full')
plt.title('Sales of Top 10 Shop')
plt.ylabel("Sales")
plt.xlabel("Month")

#figure 3
# number of items per cat 
x = data.groupby(['item_category_id'],as_index = False)['item_cnt_day'].sum()
x = x.sort_values(by='item_cnt_day',ascending=False)
x = x.iloc[0:20].reset_index()
# #plot
plt.figure(figsize=(16,8))
ax = sns.barplot(x.item_category_id, x.item_cnt_day, alpha=0.8)
plt.title("Total Sales per Category")
plt.ylabel('Sales', fontsize=12)
plt.xlabel('Category', fontsize=12)


#figure 4
x = data.groupby(['item_id','item_price'],as_index = False)['date'].count()
x = x.sort_values(by='date',ascending=False)
x = x.iloc[0:20]
top20item = x.item_id.tolist()

temp = pd.DataFrame()
for i in top20item:
    x = data.loc[(data.item_id == i)]
    temp = temp.append(x)
    
plt.figure(figsize=(16,8))
sns.boxplot(x = 'item_id', y = 'item_price',data = temp)
plt.title('price range of top 20 popular item')


#Data Preprocessing

#missing value
train.isnull().sum()
#duplicate value
train[train.duplicated(keep='first')].count()

#Outliers
plt.figure(figsize=(10,4))
plt.xlim(-200, 3000)
sns.boxplot(x = train.item_cnt_day)

plt.figure(figsize=(10,4))
plt.xlim(-20000, train.item_price.max()*1.1)
sns.boxplot(x = train.item_price)
plt.show()


