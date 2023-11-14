# Understanding Classification, Training, and Testing a Classifier

## Classification

Classification is a machine learning task where the goal is to assign predefined labels or categories to input data based on its features. It is a type of supervised learning, meaning that the algorithm learns from a labeled dataset to make predictions on new, unseen data.

## Training a Classifier

Training a classifier involves teaching a machine learning model to recognize patterns and relationships in the training data. The process typically includes the following steps:

1. **Data Collection:** Gather a dataset with labeled examples. Each example consists of input features and corresponding output labels.

2. **Data Preprocessing:** Clean and preprocess the data to ensure it is suitable for training. This may involve handling missing values, normalizing features, or encoding categorical variables.

3. **Model Selection:** Choose a classification algorithm or model based on the nature of the problem and the characteristics of the data.

4. **Feature Extraction:** Identify and select relevant features that contribute to the model's ability to make accurate predictions.

5. **Training the Model:** Use the labeled training data to train the model. During training, the algorithm adjusts its parameters to minimize the difference between predicted and actual labels.

6. **Model Evaluation:** Assess the model's performance on a separate validation set to ensure it generalizes well to new, unseen data.

7. **Hyperparameter Tuning:** Fine-tune the model's hyperparameters to optimize its performance.

## Testing a Classifier

Testing a classifier involves evaluating its performance on a separate dataset that it has not seen during training. This is crucial for assessing how well the model is expected to perform on new, real-world data. The testing process typically includes the following steps:

1. **Data Splitting:** Divide the dataset into training and testing sets. The training set is used to train the model, while the testing set is reserved for evaluating its performance.

2. **Model Prediction:** Apply the trained model to the testing set to obtain predictions for the input data.

3. **Performance Metrics:** Calculate various performance metrics such as accuracy, precision, recall, and F1 score to measure how well the model performs on the testing data.

4. **Analysis and Improvement:** Analyze the results and, if necessary, make adjustments to the model or data preprocessing steps to improve performance.

By following these steps, users can effectively develop, train, and test classifiers for various applications, ranging from image recognition to spam detection.

# Credit Card Fraud Detection Machine Learning Code

## Overview

This README provides an overview and explanation of the machine learning code used for Credit Card Fraud Detection. The code utilizes various classification algorithms to identify fraudulent transactions in a credit card dataset.

## Dataset

The datasets contains credit card transactions over a two day collection period in September 2013 by European cardholders. There are a total of 284,807 transactions, of which 492 (0.172%) are fraudulent.

The dataset contains numerical variables that are the result of a principal components analysis (PCA) transformation. This transformation was applied by the original authors to maintain confidentiality of sensitive information. Additionally the dataset contains Time and Amount, which were not transformed by PCA. The Time variable contains the seconds elapsed between each transaction and the first transaction in the dataset. The Amount variable is the transaction amount, this feature can be used for example-dependant cost-senstive learning. The Class variable is the response variable and indicates whether the transaction was fraudulant.

The dataset was collected and analysed during a research collaboration of Worldline and the Machine Learning Group of Université Libre de Bruxelles (ULB) on big data mining and fraud detection.

link of dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

## Data Exploration and Preprocessing

The initial steps involve exploring the dataset:

- Distribution plots for 'Amount' and 'Time'
- Histograms for all features
- Joint plot for 'Time' vs. 'Amount'

The code then balances the dataset by selecting a subset of non-fraudulent transactions ('Class' = 0) and combining them with all fraudulent transactions ('Class' = 1). The resulting dataset is shuffled and saved as 'creditcardsampling.csv'.

## Feature Scaling and Dimensionality Reduction

After preprocessing, the code performs feature scaling using StandardScaler on selected columns ('V22', 'V24', 'V25', 'V26', 'V27', 'V28', 'Time', 'Amount'). Next, it applies Principal Component Analysis (PCA) to reduce dimensionality, keeping only 7 principal components.

## Model Training and Evaluation

The code splits the data into training and testing sets, then proceeds with training and evaluating several classification models:

1. **Logistic Regression:**
   - Initial model and hyperparameter tuning using GridSearchCV.

   ```
   from sklearn.linear_model import LogisticRegression
   lr=LogisticRegression()
   lr.fit(X_train,y_train)
   y_pred_lr=lr.predict(X_test)

   from sklearn.metrics import classification_report,confusion_matrix
   print(confusion_matrix(y_test,y_pred_lr))

   # Hyperparamter tuning 
   from sklearn.model_selection import GridSearchCV
   lr_model = LogisticRegression()
   lr_params = {'penalty': ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
   grid_lr= GridSearchCV(lr_model, param_grid = lr_params)
   grid_lr.fit(X_train, y_train)

   grid_lr.best_params_

   y_pred_lr3=grid_lr.predict(X_test)
   print(classification_report(y_test,y_pred_lr3))
   ```

2. **Support Vector Machine (SVM):**
   - Initial model and hyperparameter tuning using GridSearchCV.

   ```
   from sklearn.svm import SVC
   svc=SVC(kernel='rbf')
   svc.fit(X_train,y_train)
   y_pred_svc=svc.predict(X_test)

   print(classification_report(y_test,y_pred_svc))

   print(confusion_matrix(y_test,y_pred_svc))

   from sklearn.model_selection import GridSearchCV
   parameters = [ {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 1, 0.01, 0.0001 ,0.001]}]
   grid_search = GridSearchCV(estimator = svc,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
   grid_search = grid_search.fit(X_train, y_train)
   best_accuracy = grid_search.best_score_
   best_parameters = grid_search.best_params_
   print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
   print("Best Parameters:", best_parameters)

   svc_param=SVC(kernel='rbf',gamma=0.01,C=100)
   svc_param.fit(X_train,y_train)
   y_pred_svc2=svc_param.predict(X_test)
   print(classification_report(y_test,y_pred_svc2))
   ```

3. **Decision Tree:**
   - Initial model and hyperparameter tuning using GridSearchCV.

   ```
   from sklearn.tree import DecisionTreeClassifier
   dtree=DecisionTreeClassifier()
   dtree.fit(X_train,y_train)
   y_pred_dtree=dtree.predict(X_test)
   print(classification_report(y_test,y_pred_dtree))

   print(confusion_matrix(y_test,y_pred_dtree))

   d_tree_param=DecisionTreeClassifier()
   tree_parameters={'criterion':['gini','entropy'],'max_depth':list(range(2,4,1)),
                 'min_samples_leaf':list(range(5,7,1))}
   grid_tree=GridSearchCV(d_tree_param,tree_parameters)
   grid_tree.fit(X_train,y_train)

   y_pred_dtree2=grid_tree.predict(X_test)

   print(classification_report(y_test,y_pred_dtree2))
   ```

4. **Random Forest:**
   - Initial model with 5 estimators.

   ```
   from sklearn.ensemble import RandomForestClassifier
   randomforest=RandomForestClassifier(n_estimators=5)
   randomforest.fit(X_train,y_train)
   y_pred_rf=randomforest.predict(X_test)
   print(confusion_matrix(y_test,y_pred_rf))

   print(classification_report(y_test,y_pred_rf))
   ```

5. **K Nearest Neighbors (KNN):**
   - Initial model and hyperparameter tuning using GridSearchCV.
   
   ```
   from sklearn.neighbors import KNeighborsClassifier
   knn=KNeighborsClassifier(n_neighbors=5)
   knn.fit(X_train,y_train)
   y_pred_knn=knn.predict(X_test)

   print(classification_report(y_test,y_pred_knn))

   print(confusion_matrix(y_test,y_pred_knn))

   knn_param=KNeighborsClassifier()
   knn_params={"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
   grid_knn=GridSearchCV(knn_param,param_grid=knn_params)
   grid_knn.fit(X_train,y_train)
   grid_knn.best_params_

   knn = KNeighborsClassifier(n_neighbors=2)

   knn.fit(X_train,y_train)
   pred_knn2 = knn.predict(X_test)

   print('WITH K=3')
   print('\n')
   print(confusion_matrix(y_test,pred_knn2))
   print('\n')
   print(classification_report(y_test,pred_knn2))
   ```

## Model Evaluation Metrics

For each model, the code calculates and prints the following evaluation metrics:

- **Confusion Matrix**
- **Classification Report**
- **Accuracy Score**
