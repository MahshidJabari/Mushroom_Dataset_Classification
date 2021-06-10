# importing libraries and methods

import os
import pandas as pd
import pydotplus
from sklearn import preprocessing
from sklearn import tree
from six import StringIO
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.tree import _tree, DecisionTreeClassifier

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

# Dataset Slicing KFold
X = balance_data.values[:, 1:23]
Y = balance_data.values[:, 0]
kf = KFold(n_splits=6)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

'''
# DecisionTree with Gini Index
clf_gini = tree.DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=20, min_samples_leaf=2)
clf_gini.fit(X_train, y_train)

# DecisionTree with ginni index output

tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=20,
                                     max_features=None, max_leaf_nodes=None, min_samples_leaf=2,
                                     min_samples_split=2, min_weight_fraction_leaf=0.0,
                                     presort=False, random_state=100, splitter='best')

# Visualizing the tree with Gini
dot_data = StringIO()
tree.export_graphviz(clf_gini, out_file=dot_data,
                     feature_names=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                                    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                                    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                                    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                                    'ring-type', 'spore-print-color', 'population', 'habitat'],
                     class_names=['e', 'p'],
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("CART.pdf")

# prediction with Gini
y_pred = clf_gini.predict(X_test)
y_pred

# CART Accuracy
print("Accuracy is ", accuracy_score(y_test, y_pred) * 100)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''
# DecisionTree with information gain
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                     max_depth=20, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=20,
                       max_features=None, max_leaf_nodes=None, min_samples_leaf=2,
                       min_samples_split=2, min_weight_fraction_leaf=0.0
                       , random_state=100, splitter='best')


# Visualizing the tree with Entropy
dot_data = StringIO()
tree.export_graphviz(clf_entropy, out_file=dot_data,
                     feature_names=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                                    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                                    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                                    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                                    'ring-type', 'spore-print-color', 'population', 'habitat'],
                     class_names=['e', 'p'],
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("ID3.pdf")

# prediction with Entropy
y_pred_en = clf_entropy.predict(X_test)
y_pred_en

# ID3 Accuracy
print("Accuracy is ", accuracy_score(y_test, y_pred_en)*100)

print(confusion_matrix(y_test, y_pred_en))
print(classification_report(y_test, y_pred_en))


our_features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                'ring-type', 'spore-print-color', 'population', 'habitat']


# Extracting rules Function
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


#tree_to_code(tree, our_features)
