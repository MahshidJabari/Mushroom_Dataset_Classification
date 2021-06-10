# importing libraries and methods
import os

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading Dataset
balance_data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
    sep=',', header=None)

# Dataset Shape
print("Dataset Lenght:: ", len(balance_data))
print("Dataset Shape:: ", balance_data.shape)

# Dataset Top Observations
print("Dataset:: ")
print(balance_data.head())

# Dataset Encoding
le = preprocessing.LabelEncoder()
balance_data = balance_data.apply(le.fit_transform)

# Dataset Slicing
X = balance_data.values[:, 1:23]
Y = balance_data.values[:, 0]

# Dataset split into train and test (HOLDOUT)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=(1 - 0.6666666666666667), random_state=100)


error = []
# Calculating error for K values between 1 and 40
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(pd.np.mean(pred_i != y_test))

classifier = KNeighborsClassifier(n_neighbors=15)
classifier.fit(X_train, y_train)

# prediction
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
