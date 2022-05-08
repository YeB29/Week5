import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("etmgeg_269.csv")
df.info()
print(df.describe())
df.isnull().sum()
df.dropna()
matrix = df.corr()
print(matrix)
# EV24      = Referentiegewasverdamping (Makkink) (in 0.1 mm) / Potential evapotranspiration (Makkink) (in 0.1 mm)
# TG        = Etmaalgemiddelde temperatuur (in 0.1 graden Celsius) / Daily mean temperature in (0.1 degrees Celsius)

X = df['EV24'].values.reshape(-1,1)
y = df['TG']


plt.scatter(X,y)
plt.xlabel('EV24')
plt.ylabel('TG')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
lr = LinearRegression()
lr.fit(X_train,y_train)

#visualiseren test data plt.scatter(X_train, y_train)
plt.plot(X_train, lr.predict(X_train))
plt.xlabel('EV24')
plt.ylabel('TG')
plt.show() 


# #visualiseren test data
plt.scatter(X_test, y_test,color='r')
plt.plot(X_test, lr.predict(X_test))
plt.xlabel('EV24')
plt.ylabel('TG')
plt.show()

# scores bepalen
print("Score train data: " + str(round(lr.score(X_train, y_train),2)))
print("Score test data: " + str(round(lr.score(X_test, y_test),2)))

# voorspelling uitvoeren
print(lr.predict([[10]]))

pickle.dump(lr, open('modelSAL.pkl','wb'))