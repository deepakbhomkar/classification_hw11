# classification_hw11
Classification HW week 11

This assignment is to build and evaluate several machine-learning models to predict credit risk using free data from LendingClub. Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans), hence we need to employ different techniques for training and evaluating models with imbalanced classes. We have used the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

*   Resampling
*   Ensemble Learning

# Resampling  

## Random Oversampling
This method shows counter for low risk and high risk as equal:  
*Counter({'low_risk': 51366, 'high_risk': 51366})*

The classification report for this model shows the following:
*   Precision score is 99% which indicates predictions were highly accurate. 
*   Recall score is 84% which indicates a high percentage of sample were identified correctly. 
*   F1 score returned was high at 91%
*   Balanced accuracy score shows 83.23 %


 
## SMOTE Oversampling
This method shows counter for low risk and high risk as equal:  
*Counter({'low_risk': 51366, 'high_risk': 51366})*

The classification report for this model shows the following:
*   Precision score is 99% which indicates predictions were highly accurate. 
*   Recall score is 87% which indicates a high percentage of sample were identified correctly This is higher than the Random Oversampling score
*   F1 score returned was high at 92% which is again higher than Random OverSampling
*   Balanced accuracy score shows 83.89 %


## Undersampling
This method shows counter for low risk and high risk as equal:  
*Counter({'high_risk': 246, 'low_risk': 246})*

The classification report for this model shows the following:
*   Precision score is 99% which indicates predictions were highly accurate. 
*   Recall score is 76% which indicates a high percentage of sample were identified correctly. This is lower than the two Oversampling models score
*   F1 score returned was high at 86% which was again lower than the two OverSampling models
*   Balanced accuracy score shows 82.16 %


##  Combination (Over and Under) sampling SMOTEENN
This method shows counter for low risk and high risk as nearly equal though not equal:  
*Counter({'high_risk': 51366, 'low_risk': 47635})*

The classification report for this model shows the following:
*   Precision score is 99% which indicates predictions were highly accurate. 
*   Recall score is 86% which indicates a high percentage of sample were identified correctly. This is 1% lower than the SMOTE Oversampling model
*   F1 score returned was high at 92% which is same as that for SMOTE Oversampling model
*   Balanced accuracy score shows 83.88 %


###  Analysis of all 4 Models  

| Model  | Balanced Accuracy Score | Precision Score | Recall Score | F1 Score |
| :--- | :---------------- | :--------- | :--------- | :------- |
| Naive Random Oversampling | 83.23 % | 99.00 % | 84.00 % | 91.00 % |
| SMOTE Oversampling | 83.89 % | 99.00 % | 87.00 % | 92.00 % |
| Undersampling | 82.16 % | 99.00 % | 76.00 % | 86.00 % |
| SMOTEENN Under & Over Sampling | 83.88 % | 99.00 % | 86.00 % | 92.00 % | 


The conclusion is SMOTE Oversampling and SMOTEENN (Combination) models have almost same scores with the exception that the SMOTE Oversampling has a 1% edge in the recall score and a 0.01 % edge in the balanced accuracy score. SMOTE Oversampling is the best model in this scenario based on the above determined scores.


# Ensemble Learning

Balanced Random Forest Classifier and Easy Ensemble Classifier models were used to test the predictions of low and high-risk credit applications. Easy Ensemble Classifier model turned out to be a better model

##  Balanced Random Forest Classifier Model
The classification report for this model shows the following:
*   Precision score is 99% which indicates predictions were highly accurate. 
*   Recall score is 87% which indicates a high percentage of sample were identified correctly. 
*   F1 score returned was high at 93%. 
*   Balanced accuracy score shows 78.88 %


##  Easy Ensemble Classifier Model
The classification report for this model shows the following:
*   Precision score is 99% which indicates predictions were highly accurate. 
*   Recall score is 94% which indicates a high percentage of sample were identified correctly. This is higher than the Balanced Random Forest Classifier model
*   F1 score returned was high at 97%. This is higher than the Balanced Random Forest Classifier model
*   Balanced accuracy score shows 83.88 % much greater than the Balanced Random Forest Classifier


The conclusion is Easy Ensemble Classifier model has better accuracy compared to Balanced Random Forest Classifier in predicting risks for credit applications.