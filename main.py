import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import plot
from sys import argv

dataset = pd.read_csv('data.csv')

# Remove all empty values
dataset = dataset.dropna()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_train_predictions = regressor.predict(X_train)

plot('Real VS. Predicted - Train set', X_train, y_train, y_train_predictions)

y_test_predictions = regressor.predict(X_test)

plot('Real VS. Predicted - Test set', X_test, y_test, y_test_predictions)

if len(argv) == 2:
    years_of_experience = int(argv[1])

    prediction = regressor.predict([[years_of_experience]])

    print(f'[+] An employee with {years_of_experience} years of experience should earn about {prediction[0]:.2f}')
else:
    print('You can specify the years of experience to get a prediction of salary by passing it as a parameter')
    print('Example: python3 ./main.py 10')
