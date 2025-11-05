 ########## SECOND DEMO #########
import numpy as np
from pysr import PySRRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Scarico il dataset
data = fetch_california_housing()
X = data.data
y = data.target


#Normalizzo 
scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = PySRRegressor(
    output_directory="output/output_demo2",
    niterations=40,
    maxsize=20,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["cos", "exp", "sin", "sqrt"],
    elementwise_loss="loss(pred, target) = (pred - target)^2",
)

#Training del modello
model.fit(X_train, y_train)

#Stampa del modello migliore
print(model.get_best())

