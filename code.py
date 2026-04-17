import pandas as pd

data= pd.read_csv('weather_dataset.csv')
print(data.head())

data['Rain']= data['Rain'].map({'Yes':1, 'No':0})

x= data[['Temperature', 'Humidity', 'WindSpeed']]
y= data['Rain']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict([[29, 68, 8]])
print("Rain Prediction:", prediction)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))