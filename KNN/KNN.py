import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distances = np.sqrt(np.sum((x1-x2)**2))
    return distances

class KNN:
    def __init__(self, k=3, lambda_reg =0.01):
        self.k=k
        self.lambda_reg = lambda_reg
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) + self.lambda_reg * np.linalg.norm(x_train)**2 for x_train in self.X_train]

        
        
        k_indicies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indicies]
        
        count_common = Counter(k_nearest_labels).most_common()
        return count_common[0][0]
         
     
