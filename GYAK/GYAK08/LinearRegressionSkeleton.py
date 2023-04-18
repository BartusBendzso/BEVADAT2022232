import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr
        self.m = 0
        self.c = 0
        self.pred = []
        self.losses = []

    def fit(self, X_train: np.array, Y_train: np.array):
        
        self.X_train = X_train
        self.Y_train = Y_train

        n = float(len(X_train)) # Number of elements in X

        # Performing Gradient Descent 
        for i in range(self.epochs): 
            Y_pred = self.m*self.X_train + self.c  # The current predicted value of Y

            residuals = self.Y_train - Y_pred # Távolságok kiszámítása
            loss = np.sum(residuals ** 2) # Ennek a távolságnak a négyzete
            self.losses.append(loss)
            D_m = (-2/n) * sum(self.X_train * residuals)  # Derivative wrt m
            D_c = (-2/n) * sum(residuals)  # Derivative wrt c
            self.m = self.m - self.lr * D_m  # Update m || kivonjuk a hibát, és itt tanul ezáltal a kódunk!
            self.c = self.c - self.lr * D_c  # Update c
            if i % 100 == 0:
                print(np.mean(self.Y_train-Y_pred))

    def predict(self, X_test):
        self.X_test = X_test
        for X in self.X_test:
            Y_pred = self.m*X + self.c # maga a prdeikció itt történik
            self.pred.append(Y_pred)

    def evaluate(self, X, Y):
        Y_test = Y
        X_test = X
        # Calculate the Mean Absolue Error
        return print("Mean Absolute Error:", np.mean(np.abs(self.pred - Y_test))), print("Mean Squared Error:", np.mean((self.pred - Y_test)**2))
        # Calculate the Mean Squared Error
        
    
    def plotter(self, Y_test):
        Y_pred = self.m*self.X_test + self.c
        plt.scatter(self.X_test, Y_test)
        plt.plot([min(self.X_test), max(self.X_test)], [min(Y_pred), max(Y_pred)], color='red') # predicted
        plt.show()
