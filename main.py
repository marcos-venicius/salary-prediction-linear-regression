import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import plot

dataset = pd.read_csv('data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_train_predictions = regressor.predict(X_train)

plot('Real VS. Predicted - Train set', X_train, y_train, y_train_predictions)

y_test_predictions = regressor.predict(X_test)

plot('Real VS. Predicted - Test set', X_test, y_test, y_test_predictions)
