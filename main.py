#importing pandas
import pandas as pd

#reading dataset
df = pd.read_csv('/content/CarPrice_Assignment.csv')
df

df.shape
df.info()

#finding maximum and minimum prices
df['price'].max()
df['price'].min()

#importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(df['price'])
#sklearn accepts only numeric values 
cars_numeric = df.select_dtypes(include = ['float64','int64']) #data cleaning 
cars_numeric

df.corr()['price'] #how much correlation each column has for the price prediction 

cars_numeric = cars_numeric.drop(['car_ID','symboling'],axis = 1)
cars_numeric.head()
cars_numeric.shape

x = cars_numeric.iloc[:,0:13].values
# x = cars_numeric.drop(columns=['price']).values

# y = cars_numeric['price'].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

cars_numeric.shape
x.shape
x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)    # (x_train,x_test,y_train,y_test)

y_pred = model.predict(x_test)
y_pred
len(y_pred)
y_test
plt.scatter(y_pred,y_test)

df1 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df1

#pandas plotting of the graphs
df1.plot(figsize=(20,10))
plt.show()

df1.plot(figsize=(20,10),kind='bar')
plt.show()

import seaborn as sns
sns.regplot(x='Actual',y='Predicted',data=df1,color='red')

#model evaluation by r2 score
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
y_test


