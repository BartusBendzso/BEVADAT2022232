import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import seaborn as sns
csv_path="diabetes.csv"

class KNNClassifier:
    
    @property
    def k_neighbors(self):
        return self.k

    def __init__(self, k:int, test_split_ratio :float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio
    
    @staticmethod 
    def load_csv(csv_path:str)-> Tuple[pd.DataFrame,pd.Series]:
        dataset = pd.read_csv(csv_path)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x,y = dataset.iloc[:,:-1],dataset.iloc[:,-1]
        return x,y
    
    
    def train_test_slpit(self, features: pd.DataFrame,
                        labels: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"
        self.x_train,self.y_train = features.iloc[:train_size,:], labels.iloc[:train_size]
        self.x_test,self.y_test = features.iloc[train_size:,:],labels.iloc[train_size:]
        
    
    def euclidean(self, element_of_x:np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((self.x_train - element_of_x)**2,axis=1))
    
    def predict(self) -> pd.Series:
        labels_pred = []
        for _, x_test_element in self.x_test.iterrows():
            distances = self.euclidean(x_test_element)
            distances = pd.concat([distances, self.y_train], axis=1)
            distances = distances.sort_values(by=0).values

            label_pred = mode(distances[:self.k,1], axis=0).mode[0]
            labels_pred.append(label_pred)
        self.y_preds = pd.Series(labels_pred, dtype=np.int64)
    
    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100

    def confusion_matrix(self) -> pd.DataFrame:
        return pd.crosstab(self.y_test, self.y_preds)

    def best_k(self) -> Tuple[int, float]:
        accuracies = []
        for k in range(1,21):
            self.k = k
            self.predict()
            accuracy = self.accuracy()
            accuracies.append((k,accuracy))
        best_k = max(accuracies, key=lambda x:x[1])
        return (best_k[0], round(best_k[1], 2))
            
