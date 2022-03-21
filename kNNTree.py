import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class kNNTree(BaseEstimator, ClassifierMixin):

    def __init__(self, k, knn_weight, min_samples_leaf=0.1, scale='feature_importance', knn_method='kd_tree'):
        self.k = k
        self.knn_weight = knn_weight
        self.scale = scale
        self.min_samples_leaf = min_samples_leaf # alternatively, we can use tree depth instead of min_samples_split
        

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        dtc = DecisionTreeClassifier(min_samples_leaf=msl)
        dtc.fit(X_train, y_train)
        self.dtc = dtc
        self.train_samples_in_leaf = self._split_samples(X_train, dtc)
        return self
        
    def _split_samples(self, X, tree_classifier):
        leaf_indices = tree_classifier.apply(X) # length equal to number of samples, each entry is a leaf index
        dict_samples_in_leaf = dict([(key, []) for key in list(set(leaf_indices))])
        for l in range(len(leaf_indices)):
            dict_samples_in_leaf[leaf_indices[l]].append(l)
        return dict_samples_in_leaf
        
    def _predict_leaf(self, X_subtest, leaf_node):
        k = self.k
        dtc = self.dtc
        subtrain_indices = self.train_samples_in_leaf[leaf_node]
        X_subtrain = self.X_train[subtrain_indices]
        y_subtrain = self.y_train[subtrain_indices]
        
        if (len(X_subtrain) <= k) or (np.min(dtc.value[leaf_node]) <= int(k/2)):
                c = dtc.classes_[np.argmax(dtc.value[leaf_node])]
                y_subpred = [c for i in range(len(X_subtest))]
        else:
            knc = KNeighborsClassifier(n_neighbors = k, algorithm=knn_method, weights=knn_weight)
            if self.scale == 'feature_importance':
                fi = dtc.feature_importances_
                knc.fit(np.sqrt(fi)*X_subtrain, y_subtrain)
                y_subpred  = knc.predict(np.sqrt(fi)*X_subtest)
            else:
                knc.fit(X_subtrain, y_subtrain)
                y_subpred  = knc.predict(X_subtest)
        return y_subpred
    
    def predict(self, X_test):
        
        dtc = self.dtc
        
        train_leaf = self.train_samples_in_leaf
        test_leaf = self._split_samples(X_test, dtc)
        
        leaf_indices = train_leaf.keys()
        
        y_pred = np.zeros(X_test.shape[0],)
        
        for leaf_node in leaf_indices:
            subtest_indices = test_leaf[leaf_node]
            y_pred[subtest_indices] = self._predict_leaf(X_test[subtest_indices], leaf_node)
        
        return y_pred
            
        
        
        
        
    
    
