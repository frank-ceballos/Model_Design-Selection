""" ***************************************************************************
# * File Description:                                                         *
# * Pipeline for model building                                               *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Define class                                                           *
# * 3. Get data                                                               *
# * 4. Create train and test set                                              *
# * 5. Feature scaling                                                        *
# * 6. Parameter tuning                                                       *
# * 7. Feature selection                                                      *
# * 8. Build model with selected features                                     *
$ * 9. Cast and evaluate model predictions                                    *
# * 10. Visualize results                                                     *
# *                                                                           *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Frank Ceballos <frank.ceballos89@gmail.com>                   *
# * --------------------------------------------------------------------------*
# * DATE CREATED: June 26, 2019                                               *
# * --------------------------------------------------------------------------*
# * NOTES: None                                                               *
# * ************************************************************************"""


###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

###############################################################################
#                          2. Importing Libraries                             #
###############################################################################
class RFECV:
    """
    Features are ranked using recursive feature elimination with cross validation.
    
    Parameters
    ----------
    estimator: estimator object
    A Scikit-learn estimator with a feature_importances_ attribute.
        
    cv : int, cross-validation generator or an iterable, optional
    
    
    Attributes
    ----------
    ranking: dict
    Dictionary containing feature labels, ranking, and score.  For each iteration,
    the score is the average of all the folds. 
    
    """
    
    def __init__(self, estimator, cv):
        self.estimator = estimator
        self.cv = cv
        
        # Initiate variables
        self.ranking = pd.DataFrame(columns = ["Feature Label", "Rank No",
                                               "Score"])
    
    
    def fit(self, X_train, y_train):
        self.rank_features(X_train, y_train)
    
    
    def rank_features(self, X_train, y_train):
        # Initialize variables and dictionaries
        n_features = len(X_train.columns)
        ranked_features = {"Feature Label": [],
                           "Feature Rank": [],
                           "Feature Score": []}
        
        # k-folds cross-validator
        kf = KFold(n_splits = self.cv)
        
        for feature_no in range(n_features):
            # Starting features
            if feature_no == 0:
                feature_labels = list(X_train.columns)
                
            # Get X_train with surving features
            X_train = X_train[feature_labels]
            
            # Cross validaded score and importances
            cv_score = 0
            importances = np.zeros(len(feature_labels))

            print(f"Now making model with {n_features - feature_no} features")
            for train_index, eval_index in kf.split(X_train):   
                
                # Train estimator
                self.estimator.fit(X_train.iloc[train_index, :], y_train[train_index])
                
                # Make predictions
                y_pred = self.estimator.predict_proba(X_train.iloc[eval_index, :])[:,1]
                
                # Evaluate estimator
                eval_score = metrics.roc_auc_score(y_train[eval_index], y_pred)
                
                # Accumulate eval_score
                cv_score += eval_score
            
                # Accumulate feature importance
                importances += self.estimator.feature_importances_
                                    
            
            # Determine less important feature
            to_drop_index = np.argmin(importances)
            to_drop_label = feature_labels[to_drop_index]
            
            # Save results
            ranked_features["Feature Label"] = ranked_features["Feature Label"] + [to_drop_label]
            ranked_features["Feature Rank"] = ranked_features["Feature Rank"] + [n_features - feature_no]
            ranked_features["Feature Score"] = ranked_features["Feature Score"] + [cv_score/self.cv]
            
            # Remove less important feature
            feature_labels = [label for label in feature_labels if label != to_drop_label]
        
        # Save all results
        self.ranking = pd.DataFrame(ranked_features)
                
                        

###############################################################################
#                                 3. Get data                                 #
###############################################################################
X, y = make_classification(n_samples = 2000, n_features = 30, n_redundant = 15,
                           n_informative = 5, n_repeated = 0, 
                           n_clusters_per_class = 2, class_sep = 0.75,
                           random_state = 1000)

labels = [f"Feature {ii+1}" for ii in range(X.shape[1])]
X = pd.DataFrame(X, columns = labels)
y = pd.DataFrame(y, columns = ["Target"])

# Numpy array to pandas dataframe
labels = [f"Feature {ii+1}" for ii in range(X_train.shape[1])]
X_train = pd.DataFrame(X_train, columns = labels)
X_test = pd.DataFrame(X_test, columns = labels)

###############################################################################
#                       4. Create train and test set                          #
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                    random_state = 1000)



###############################################################################
#                     5. Removing highly correlated features                  #
###############################################################################
# Filter Method: Spearman's Cross Correlation > 0.95
# Make correlation matrix
corr_matrix = X_train.corr(method = "spearman").abs()

# Draw the heatmap
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, cmap= "YlGnBu", square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax)
f.tight_layout()

# Select upper triangle of matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
X_train = X_train.drop(to_drop, axis = 1)



###############################################################################
#                           6. Feature Processing                             #
###############################################################################
# Scale features via Z-score normalization
scaler = StandardScaler()
rfecv = RFECV(estimator=RandomForestClassifier(), cv = 5)
steps = [("scaler", scaler)]
feature_processing = Pipeline(steps = steps)






###############################################################################
#                             6. Parameter tuning                             #
###############################################################################
classifier = RandomForestClassifier()

params = {"n_jobs": [-1],
          "n_estimators": [200],
          "criterion": ["gini", "entropy"],
          "min_samples_leaf": [0.05, 0.075, 0.1],
          "min_samples_split": [0.05, 0.075, 0.1],
          "class_weight": [None, "balanced", "balanced_subsample"]}

grid = GridSearchCV(estimator = classifier, param_grid = params, 
                    scoring = "roc_auc", cv = 5, verbose = 50)

grid.fit(X_train, y_train)

# Get tuned parameters
tuned_params = grid.best_params_

# Set tuned parameters
classifier.set_params(**tuned_params)


###############################################################################
#                          7. Feature selection                               #
###############################################################################
# Filter Method: Spearman's Cross Correlation > 0.95
# Make correlation matrix
corr_matrix = X_train.corr(method = "spearman").abs()

# Draw the heatmap
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, cmap= "YlGnBu", square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax)
f.tight_layout()

# Select upper triangle of matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
X_train = X_train.drop(to_drop, axis = 1)



# Wrapper Method: Recursive Feature Elimination with k-fold cross validation
classifier.n_estimators = 1000
rfecv = RFECV(classifier, 5)
rfecv.fit(X_train, y_train)


sns.lineplot(x = "Feature Rank", y = "Feature Score", data = rfecv.ranking)