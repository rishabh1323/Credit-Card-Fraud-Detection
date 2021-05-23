# Credit Card Fraud Detection
Dataset Link - https://www.kaggle.com/mlg-ulb/creditcardfraud

- The datasets contains transactions made by credit cards in September 2013 by European cardholders.
- This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
- It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the author cannot provide the original features and more background information about the data. 
- Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 
- Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Exploratory Data Analysis

#### Importing Libraries and Dataset


```python
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

%matplotlib inline
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
```


```python
# Importing the dataset
df = pd.read_csv('creditcard.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>0.090794</td>
      <td>-0.551600</td>
      <td>-0.617801</td>
      <td>-0.991390</td>
      <td>-0.311169</td>
      <td>1.468177</td>
      <td>-0.470401</td>
      <td>0.207971</td>
      <td>0.025791</td>
      <td>0.403993</td>
      <td>0.251412</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>-0.166974</td>
      <td>1.612727</td>
      <td>1.065235</td>
      <td>0.489095</td>
      <td>-0.143772</td>
      <td>0.635558</td>
      <td>0.463917</td>
      <td>-0.114805</td>
      <td>-0.183361</td>
      <td>-0.145783</td>
      <td>-0.069083</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>0.207643</td>
      <td>0.624501</td>
      <td>0.066084</td>
      <td>0.717293</td>
      <td>-0.165946</td>
      <td>2.345865</td>
      <td>-2.890083</td>
      <td>1.109969</td>
      <td>-0.121359</td>
      <td>-2.261857</td>
      <td>0.524980</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>-0.054952</td>
      <td>-0.226487</td>
      <td>0.178228</td>
      <td>0.507757</td>
      <td>-0.287924</td>
      <td>-0.631418</td>
      <td>-1.059647</td>
      <td>-0.684093</td>
      <td>1.965775</td>
      <td>-1.232622</td>
      <td>-0.208038</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>0.753074</td>
      <td>-0.822843</td>
      <td>0.538196</td>
      <td>1.345852</td>
      <td>-1.119670</td>
      <td>0.175121</td>
      <td>-0.451449</td>
      <td>-0.237033</td>
      <td>-0.038195</td>
      <td>0.803487</td>
      <td>0.408542</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Printing the dimanesions of the dataframe
df.shape
```




    (284807, 31)



#### Checking for Missing/NaN Values


```python
# Printing number of missing/NaN values in each feature
df.isnull().sum()
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64



- We can see that there are no NaN values present in the dataset, so no need to worry about them.

#### Visualizing Independent Features


```python
# Extracting independent features
independent_features = df.columns[:-1]
independent_features = list(independent_features)
print(independent_features)
```

    ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    


```python
# Plotting histograms for all the independent features with their kernel density estimations
fig, axes = plt.subplots(nrows=15, ncols=2, figsize=(7*2, 5*15))

for i, feature in enumerate(independent_features):
    row = i // 2
    col = i % 2
    sns.histplot(x=df[feature], data=df, kde=True, ax=axes[row][col])
    axes[row][col].set_title(f'Histogram for {feature}')
```


    
![png](outputs/output_12_0.png)
    



```python
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
```


```python
# Plotting Q-Q plots for the independent features
for feature in independent_features:
    plot_QQ(df, feature)
```


    
![png](outputs/output_14_0.png)
    



    
![png](outputs/output_14_1.png)
    



    
![png](outputs/output_14_2.png)
    



    
![png](outputs/output_14_3.png)
    



    
![png](outputs/output_14_4.png)
    



    
![png](outputs/output_14_5.png)
    



    
![png](outputs/output_14_6.png)
    



    
![png](outputs/output_14_7.png)
    



    
![png](outputs/output_14_8.png)
    



    
![png](outputs/output_14_9.png)
    



    
![png](outputs/output_14_10.png)
    



    
![png](outputs/output_14_11.png)
    



    
![png](outputs/output_14_12.png)
    



    
![png](outputs/output_14_13.png)
    



    
![png](outputs/output_14_14.png)
    



    
![png](outputs/output_14_15.png)
    



    
![png](outputs/output_14_16.png)
    



    
![png](outputs/output_14_17.png)
    



    
![png](outputs/output_14_18.png)
    



    
![png](outputs/output_14_19.png)
    



    
![png](outputs/output_14_20.png)
    



    
![png](outputs/output_14_21.png)
    



    
![png](outputs/output_14_22.png)
    



    
![png](outputs/output_14_23.png)
    



    
![png](outputs/output_14_24.png)
    



    
![png](outputs/output_14_25.png)
    



    
![png](outputs/output_14_26.png)
    



    
![png](outputs/output_14_27.png)
    



    
![png](outputs/output_14_28.png)
    



    
![png](outputs/output_14_29.png)
    


- We can see that most of the features are roughly normally distributed, but they have a lot of outliers.
- We can try to fix this by applying some transformations.

#### Visualizing Dependent Feature


```python
# Plotting countplot of the dependent feature
plt.figure(figsize=(8, 6))
sns.countplot(x=df['Class'])
plt.title('Countplot for Dependent Feature \'Class\'')
plt.show()
```


    
![png](outputs/output_17_0.png)
    



```python
# Printing number of records for each class in 'Class'
df['Class'].value_counts()
```




    0    284315
    1       492
    Name: Class, dtype: int64



- We can see that the dependent feature 'Class' is highly imbalanced, where number of instances of class 0 (genuine or not-fraud transaction) is much much more than that of class 1.
- We would need to apply some kind of feature engineering technique to balance the dependent feature.

#### Visualizing Relationship Between Independent Features and Dependent Feature


```python
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
```


    
![png](outputs/output_21_0.png)
    



```python
# Plotting a boxplot for each independent feature against the dependent feature
fig, axes = plt.subplots(10, 3, figsize=(20, 40))

for i, feature in enumerate(independent_features):
    row = i // 3
    col = i % 3
    
    sns.boxplot(x=df['Class'], y=df[feature], ax=axes[row][col])
    axes[row][col].set_title(f'Class vs {feature}')
    
    plt.tight_layout(pad=0.3)
```


    
![png](outputs/output_22_0.png)
    


#### Plotting Correlation Heatmap


```python
# Plotting correlation heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
plt.title('Heatmap for Correlation Matrix')
plt.show()
```


    
![png](outputs/output_24_0.png)
    


- From the correlation heatmap we can clearly see that non of the features have high correlation.

## Feature Selection

#### Splitting the Data into Train and Test Sets


```python
# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[independent_features], df['Class'], test_size=0.2, random_state=0)
print('Dimensions of X_train :', X_train.shape)
print('Dimensions of y_train :', y_train.shape)
print('Dimensions of X_test  :', X_test.shape)
print('Dimensions of y_test  :', y_test.shape)
```

    Dimensions of X_train : (227845, 30)
    Dimensions of y_train : (227845,)
    Dimensions of X_test  : (56962, 30)
    Dimensions of y_test  : (56962,)
    

#### Feature Importance using Extra Tree Classifier


```python
# Training ExtraTreeClassifier model on train data and plotting the feature importances
extra_tree_classifier = ExtraTreesClassifier()
extra_tree_classifier.fit(X_train, y_train)

feature_importances = pd.Series(extra_tree_classifier.feature_importances_, X_train.columns).sort_values()
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=feature_importances.index, orient='h')
plt.title('Feature Importances based on ExtraTreeRegressor')
plt.xlabel('Feature Importance')
plt.show()
```


    
![png](outputs/output_30_0.png)
    



```python
# Extracting the top 10 independent features
features_selected = list(feature_importances.index[-10:])
print('Top 10 Most Important Features are\n\n', features_selected)
```

    Top 10 Most Important Features are
    
     ['V9', 'V3', 'V18', 'V4', 'V10', 'V11', 'V12', 'V16', 'V14', 'V17']
    


```python
# Keeping only the selected features
X_train = X_train[features_selected]
X_test = X_test[features_selected]
```

## Feature Engineering

#### Simultaneous Upsampling and Downsampling using SMOTETomek


```python
# Resampling the imbalanced training set to have a more balanced distribution of the dependent feature 'Class'
print('Number of examples in dependent feature before resampling', Counter(y_train))
sampler = SMOTETomek(random_state=0)
X_train, y_train = sampler.fit_resample(X_train, y_train)
print('Number of examples in dependent feature after resampling', Counter(y_train))
```

    Number of examples in dependent feature before resampling Counter({0: 227454, 1: 391})
    Number of examples in dependent feature after resampling Counter({0: 227454, 1: 227454})
    

## Model Building

#### Logistic Regression Classifier


```python
# Defining hyperparameter values to tune over
grid_params = {
    'C' : 10.0 ** np.arange(-2, 2),
     'penalty' : ['l1', 'l2'],
    'class_weight' : [{0:1, 1:10}, {0:1, 1:100}, {0:1, 1:1000}, {0:1, 1:10000}]
}
```


```python
# Building a Logistic Regression Classifier model 
logistic_classifier = GridSearchCV(estimator=LogisticRegression(n_jobs=-1), param_grid=grid_params, 
                                   cv=3, scoring='f1_macro', n_jobs=-1)
logistic_classifier.fit(X_train, y_train)
```




    GridSearchCV(cv=3, estimator=LogisticRegression(n_jobs=-1), n_jobs=-1,
                 param_grid={'C': array([ 0.01,  0.1 ,  1.  , 10.  ]),
                             'class_weight': [{0: 1, 1: 10}, {0: 1, 1: 100},
                                              {0: 1, 1: 1000}, {0: 1, 1: 10000}],
                             'penalty': ['l1', 'l2']},
                 scoring='f1_macro')




```python
# Printing best hyperparamters
print('Best hyperparameters:', logistic_classifier.best_params_)

# Creating final logistic regression model with best hyperparameteres
logistic_classifier = LogisticRegression(C=0.01, class_weight={0:1,1:10}, penalty='l2', n_jobs=-1)
logistic_classifier.fit(X_train, y_train)
```

    Best hyperparameters: {'C': 0.1, 'class_weight': {0: 1, 1: 10}, 'penalty': 'l2'}
    




    LogisticRegression(C=0.01, class_weight={0: 1, 1: 10}, n_jobs=-1)




```python
# Making predictions on test data and printing metrics for those predictions
y_pred = logistic_classifier.predict(X_test)
print('Accuracy : ', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix\n\n', confusion_matrix(y_test, y_pred))
print('\nClassification Report\n\n', classification_report(y_test, y_pred))
```

    Accuracy :  0.7785014571117587
    
    Confusion Matrix
    
     [[44247 12614]
     [    3    98]]
    
    Classification Report
    
                   precision    recall  f1-score   support
    
               0       1.00      0.78      0.88     56861
               1       0.01      0.97      0.02       101
    
        accuracy                           0.78     56962
       macro avg       0.50      0.87      0.45     56962
    weighted avg       1.00      0.78      0.87     56962
    
    

- We can see that the precision, recall and f1-score for class 1 are low.
- The number of false negatives is very high, that is, the model fails to identifies many fraud transactions.
- This means the model is not performing well on the data.
- Even though the accuracy is expectionally good, it is not a good metric as the dataset is highly imbalanced.

#### Random Forest Classifier


```python
# Creating random forest model with default hyperparameteres
random_forest_classifier = RandomForestClassifier(n_jobs=-1)
random_forest_classifier.fit(X_train, y_train)
```




    RandomForestClassifier(n_jobs=-1)




```python
# Making predictions on test data and printing metrics for those predictions
y_pred = random_forest_classifier.predict(X_test)
print('Accuracy : ', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix\n\n', confusion_matrix(y_test, y_pred))
print('\nClassification Report\n\n', classification_report(y_test, y_pred))
```

    Accuracy :  0.9993504441557529
    
    Confusion Matrix
    
     [[56840    21]
     [   16    85]]
    
    Classification Report
    
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     56861
               1       0.80      0.84      0.82       101
    
        accuracy                           1.00     56962
       macro avg       0.90      0.92      0.91     56962
    weighted avg       1.00      1.00      1.00     56962
    
    

- We can see that the precision, recall and f1-score values are pretty good
- The number of false negatives is quite low, but maybe we can try to reduce them even further
- Till now we can confidently say that only ensemble models are performing well given that we have not performed any feature engineering

#### Dumping Models to Pickle File


```python
# Creating a pickle file and dumping all models to it
with open('models.pkl', 'wb') as file:
    pickle.dump([logistic_classifier, random_forest_classifier], file)
```
