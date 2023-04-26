import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

class KMeansOnDigits():
    
    def __init__(self, n_clusters: int, random_state: int) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def load_dataset(self):
        self.digits = load_digits()

    def predict(self) -> tuple:
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.clusters = kmeans.fit_predict(self.digits.data)    

    def get_labels(self):
        result_array = np.ndarray(shape = len(self.clusters))
        amount_of_clusters = len(np.unique(self.clusters))
        for cluster in range(amount_of_clusters):
            mask = self.clusters == cluster
            label = np.bincount(self.digits.target[mask]).argmax()
            result_array[mask] = label
        self.labels = result_array

    def calc_accuracy(self) -> float:
        accuracy_sc = accuracy_score(self.digits.target, self.labels)
        self.accuracy = round(accuracy_sc,2)

    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)
    