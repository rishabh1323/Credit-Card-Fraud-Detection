# Importing required libraries
import pylab
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from collections import Counter
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Importing the dataset
df = pd.read_csv('creditcard.csv')
print(df.head())

# Printing the dimanesions of the dataframe
print(df.shape)

# Printing number of missing/NaN values in each feature
print(df.isnull().sum())

# Extracting independent features
independent_features = df.columns[:-1]
independent_features = list(independent_features)
print(independent_features)

# Plotting histograms for all the independent features with their kernel density estimations
fig, axes = plt.subplots(nrows=15, ncols=2, figsize=(7*2, 5*15))

for i, feature in enumerate(independent_features):
    row = i // 2
    col = i % 2
    sns.histplot(x=df[feature], data=df, kde=True, ax=axes[row][col])
    axes[row][col].set_title(f'Histogram for {feature}')

# Defining a function to plot Q-Q plots
def plot_QQ(df, feature):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(x=df[feature], kde=True)
    plt.title(f'Histogram for {feature}')
    
    plt.subplot(1, 2, 2)
    stats.probplot(x=df[feature], dist='norm', plot=pylab)
    plt.title(f'Probably Plot for {feature}')
    
    plt.tight_layout()
    plt.show()

# Plotting Q-Q plots for the independent features
for feature in independent_features:
    plot_QQ(df, feature)

# Plotting countplot of the dependent feature
plt.figure(figsize=(8, 6))
sns.countplot(x=df['Class'])
plt.title('Countplot for Dependent Feature \'Class\'')
plt.show()

# Printing number of records for each class in 'Class'
print(df['Class'].value_counts())

# Plotting a scatterplot for each independent feature against the dependent feature
fig, axes = plt.subplots(5, 6, figsize=(20, 12))

for i, feature in enumerate(independent_features):
    row = i // 6
    col = i % 6
    
    axes[row][col].scatter(x=df['Class'], y=df[feature])
    axes[row][col].set_xticks((0, 1))
    axes[row][col].set_title(f'{feature} vs Class')
    axes[row][col].set_xlabel('Class')
    axes[row][col].set_ylabel(feature)
    
    plt.tight_layout(pad=0.3)

# Plotting a boxplot for each independent feature against the dependent feature
fig, axes = plt.subplots(10, 3, figsize=(20, 40))

for i, feature in enumerate(independent_features):
    row = i // 3
    col = i % 3
    
    sns.boxplot(x=df['Class'], y=df[feature], ax=axes[row][col])
    axes[row][col].set_title(f'Class vs {feature}')
    
    plt.tight_layout(pad=0.3)

# Plotting correlation heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
plt.title('Heatmap for Correlation Matrix')
plt.show()

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[independent_features], df['Class'], test_size=0.2, random_state=0)
print('Dimensions of X_train :', X_train.shape)
print('Dimensions of y_train :', y_train.shape)
print('Dimensions of X_test  :', X_test.shape)
print('Dimensions of y_test  :', y_test.shape)

# Training ExtraTreeClassifier model on train data and plotting the feature importances
extra_tree_classifier = ExtraTreesClassifier()
extra_tree_classifier.fit(X_train, y_train)

feature_importances = pd.Series(extra_tree_classifier.feature_importances_, X_train.columns).sort_values()
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=feature_importances.index, orient='h')
plt.title('Feature Importances based on ExtraTreeRegressor')
plt.xlabel('Feature Importance')
plt.show()

# Extracting the top 10 independent features
features_selected = list(feature_importances.index[-10:])
print('Top 10 Most Important Features are\n\n', features_selected)

# Keeping only the selected features
X_train = X_train[features_selected]
X_test = X_test[features_selected]

# Resampling the imbalanced training set to have a more balanced distribution of the dependent feature 'Class'
print('Number of examples in dependent feature before resampling', Counter(y_train))
sampler = SMOTETomek(random_state=0)
X_train, y_train = sampler.fit_resample(X_train, y_train)
print('Number of examples in dependent feature after resampling', Counter(y_train))

# Defining hyperparameter values to tune over
grid_params = {
    'C' : 10.0 ** np.arange(-2, 2),
     'penalty' : ['l1', 'l2'],
    'class_weight' : [{0:1, 1:10}, {0:1, 1:100}, {0:1, 1:1000}, {0:1, 1:10000}]
}

# Building a Logistic Regression Classifier model 
logistic_classifier = GridSearchCV(estimator=LogisticRegression(n_jobs=-1), param_grid=grid_params, 
                                   cv=3, scoring='f1_macro', n_jobs=-1)
logistic_classifier.fit(X_train, y_train)

# Making predictions on test data and printing metrics for those predictions
y_pred_logistic = logistic_classifier.predict(X_test)
print('Accuracy : ', accuracy_score(y_test, y_pred_logistic))
print('\nConfusion Matrix\n\n', confusion_matrix(y_test, y_pred_logistic))
print('\nClassification Report\n\n', classification_report(y_test, y_pred_logistic))

# Printing best parameters
print(logistic_classifier.best_params_)

# Defining hyperparameter values to tune over
grid_params = {
    'n_estimators' : [100, 250, 500],
    'class_weight' : [{0:1, 1:1}, {0:1, 1:100}, {0:1, 1:1000}]
}

# Building a Random Forest Classifier model 
random_forest_classifier = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1), param_grid=grid_params, 
                                        cv=3, scoring='f1_macro', n_jobs=-1)
random_forest_classifier.fit(X_train, y_train)

# Making predictions on test data and printing metrics for those predictions
y_pred_random_forest = logistic_classifier.predict(X_test)
print('Accuracy : ', accuracy_score(y_test, y_pred_random_forest))
print('\nConfusion Matrix\n\n', confusion_matrix(y_test, y_pred_random_forest))
print('\nClassification Report\n\n', classification_report(y_test, y_pred_random_forest))

# Printing best parameters
print(random_forest_classifier.best_params_)

# Creating a pickle file and dumping all 4 models to it
pickle_file = open('models.pkl', 'wb')
pickle.dump([logistic_classifier, random_forest_classifier], pickle_file)
pickle_file.close()