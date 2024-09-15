# CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
## OVERVIEW
This repository contains a machine learning project that includes the implementation of 12 different classifiers. The primary goal of the project is to evaluate the performance of various classifiers on a given dataset and determine the best performing model. The Random Forest classifier achieved the highest test accuracy..
## ABSTRACT
As credit card transactions grow globally, the risk of fraud increases, posing significant challenges to the financial industry. Machine learning offers promising solutions for detecting credit card fraud in both online and offline transactions. This project compares 12 classifiers, including Logistic Regression, KNN, SVM, Decision Tree, Random Forest, and others, on the Credit Card Fraud Detection dataset from Kaggle. The dataset, with its highly imbalanced nature, is addressed using a combination of undersampling and oversampling techniques to improve model performance.  The dataset of credit card transactions obtained from Kaggle containing 284,807 transactions. A mixture of under-sampling and oversampling techniques applied to the unbalanced data. The five strategies used to the raw and preprocessed data, respectively. This work implemented in Python.The Random Forest classifier achieved the highest test accuracy.
## ACKNOWLEDGEMENT
 I would like sincerely to thank the author from the Kaggle platform which offers the dataset. The dataset has been collected and analyzed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB on big data mining and fraud detection. More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the Defeat Fraud project.
## DATASET
The dataset contains transactions made by credit cards in September 2013 by European cardholders.This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
## DATASET LINK
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
## METHODOLOGY
The following steps were followed in the project:
1. **Data Preprocessing**:
2. Handling missing values, normalizing the `Amount` column, and splitting the dataset into training and testing sets.
3. **Modeling**:
4. Used 12 different classifiers to evaluate their performance on detecting fraud:
   - Adaboost
   - ANN(Artificial Neural Networks)
   - Decision Tree
   - Gradient Boosting
   - K- Nearest Neighbour
   - Linear Regression
   - Logistic Regression
   - MLP
   - Naive Bayes
   - Random Forest
   - SVM (Support Vector Machines)
   - XG Booster
5. **Model Evaluation**:
6. Accuracy, precision, recall, F1-score, and ROC-AUC were calculated to evaluate the performance of each classifier.

## CLASSIFIER PERFORMANCE
Here is a summary of the performance of each classifier evaluated in this project:

| Classifier                      | Train Accuracy | Test Accuracy |
|---------------------------------|----------------|---------------|
| AdaBoost                        |     0.99       |     0.89      |
| ANN (Artificial Neural Network) |     0.99       |     0.89      |
| Decision Tree                   |     0.96       |     0.91      |
| Gradient Boosting               |     0.99       |     0.92      |
| K-Nearest Neighbors (KNN)       |     0.94       |     0.92      |
| Linear Regression               |     0.92       |     0.89      |
| Logistic Regression             |     0.95       |     0.93      |
| MLP (Multi-Layer Perceptron)    |     0.68       |     0.68      |
| Naive Bayes                     |     0.90       |     0.90      |
| Random Forest                   |     0.96       |     0.95      |
| SVM (Support Vector Machine)    |     0.91       |     0.90      |
| XGBoost                         |     0.99       |     0.92      |

## EXPERIMENTAL RESULTS AND DISCUSSION
Based on the updated results, the Random Forest classifier now exhibits the highest accuracy on the testing dataset (95%), surpassing all other models. However, it's crucial to consider additional metrics such as precision, recall, and F1-score for a comprehensive evaluation of model performance. Despite its high accuracy, we observe that Random Forest may not be the optimal choice if detecting all instances of fraud is crucial, as it may have lower recall compared to other models.

Logistic Regression still maintains a good balance between precision, recall, and accuracy, making it a strong candidate for credit card fraud detection in this scenario. While Random Forest shows promising results, further fine-tuning and testing on larger datasets are necessary to confirm its effectiveness in real-world applications.
