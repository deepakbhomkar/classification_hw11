```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
```


```python
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
```

# Read the CSV and Perform Basic Data Cleaning


```python
# https://help.lendingclub.com/hc/en-us/articles/215488038-What-do-the-different-Note-statuses-mean-

columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]

target = ["loan_status"]
```


```python
# Load the data
file_path = Path('../Resources/LoanStats_2019Q1.csv.zip')
df = pd.read_csv(file_path, skiprows=1)[:-2]
df = df.loc[:, columns].copy()

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')

# Drop the null rows
df = df.dropna()

# Remove the `Issued` loan status
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# convert interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100


# Convert the target column values to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}   
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)

df.reset_index(inplace=True, drop=True)

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
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>dti</th>
      <th>...</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
      <th>hardship_flag</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10500.0</td>
      <td>0.1719</td>
      <td>375.35</td>
      <td>RENT</td>
      <td>66000.0</td>
      <td>Source Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>27.24</td>
      <td>...</td>
      <td>85.7</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>65687.0</td>
      <td>38199.0</td>
      <td>2000.0</td>
      <td>61987.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25000.0</td>
      <td>0.2000</td>
      <td>929.09</td>
      <td>MORTGAGE</td>
      <td>105000.0</td>
      <td>Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>20.23</td>
      <td>...</td>
      <td>91.2</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>271427.0</td>
      <td>60641.0</td>
      <td>41200.0</td>
      <td>49197.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000.0</td>
      <td>0.2000</td>
      <td>529.88</td>
      <td>MORTGAGE</td>
      <td>56000.0</td>
      <td>Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>24.26</td>
      <td>...</td>
      <td>66.7</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60644.0</td>
      <td>45684.0</td>
      <td>7500.0</td>
      <td>43144.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>0.1640</td>
      <td>353.55</td>
      <td>RENT</td>
      <td>92000.0</td>
      <td>Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>31.44</td>
      <td>...</td>
      <td>100.0</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>99506.0</td>
      <td>68784.0</td>
      <td>19700.0</td>
      <td>76506.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22000.0</td>
      <td>0.1474</td>
      <td>520.39</td>
      <td>MORTGAGE</td>
      <td>52000.0</td>
      <td>Not Verified</td>
      <td>Mar-2019</td>
      <td>low_risk</td>
      <td>n</td>
      <td>18.76</td>
      <td>...</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>219750.0</td>
      <td>25919.0</td>
      <td>27600.0</td>
      <td>20000.0</td>
      <td>N</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>



# Split the Data into Training and Testing


```python
# Binary encoding using Pandas (multiple columns)
df_binary_encoded = pd.get_dummies(df, columns=["home_ownership", "verification_status", 
                                                "issue_d", "pymnt_plan", "initial_list_status", 
                                                "next_pymnt_d", "application_type", 
                                                "hardship_flag", "debt_settlement_flag"])
df_binary_encoded.head()
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
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>loan_status</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>...</th>
      <th>issue_d_Mar-2019</th>
      <th>pymnt_plan_n</th>
      <th>initial_list_status_f</th>
      <th>initial_list_status_w</th>
      <th>next_pymnt_d_Apr-2019</th>
      <th>next_pymnt_d_May-2019</th>
      <th>application_type_Individual</th>
      <th>application_type_Joint App</th>
      <th>hardship_flag_N</th>
      <th>debt_settlement_flag_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10500.0</td>
      <td>0.1719</td>
      <td>375.35</td>
      <td>66000.0</td>
      <td>low_risk</td>
      <td>27.24</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25000.0</td>
      <td>0.2000</td>
      <td>929.09</td>
      <td>105000.0</td>
      <td>low_risk</td>
      <td>20.23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000.0</td>
      <td>0.2000</td>
      <td>529.88</td>
      <td>56000.0</td>
      <td>low_risk</td>
      <td>24.26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>0.1640</td>
      <td>353.55</td>
      <td>92000.0</td>
      <td>low_risk</td>
      <td>31.44</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22000.0</td>
      <td>0.1474</td>
      <td>520.39</td>
      <td>52000.0</td>
      <td>low_risk</td>
      <td>18.76</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 96 columns</p>
</div>




```python
# Create our target
y = df_binary_encoded["loan_status"]

# Create our features
X = df_binary_encoded.drop(columns="loan_status")
```


```python
X.describe()
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
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>...</th>
      <th>issue_d_Mar-2019</th>
      <th>pymnt_plan_n</th>
      <th>initial_list_status_f</th>
      <th>initial_list_status_w</th>
      <th>next_pymnt_d_Apr-2019</th>
      <th>next_pymnt_d_May-2019</th>
      <th>application_type_Individual</th>
      <th>application_type_Joint App</th>
      <th>hardship_flag_N</th>
      <th>debt_settlement_flag_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>6.881700e+04</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>...</td>
      <td>68817.000000</td>
      <td>68817.0</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.0</td>
      <td>68817.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16677.594562</td>
      <td>0.127718</td>
      <td>480.652863</td>
      <td>8.821371e+04</td>
      <td>21.778153</td>
      <td>0.217766</td>
      <td>0.497697</td>
      <td>12.587340</td>
      <td>0.126030</td>
      <td>17604.142828</td>
      <td>...</td>
      <td>0.177238</td>
      <td>1.0</td>
      <td>0.123879</td>
      <td>0.876121</td>
      <td>0.383161</td>
      <td>0.616839</td>
      <td>0.860340</td>
      <td>0.139660</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10277.348590</td>
      <td>0.048130</td>
      <td>288.062432</td>
      <td>1.155800e+05</td>
      <td>20.199244</td>
      <td>0.718367</td>
      <td>0.758122</td>
      <td>6.022869</td>
      <td>0.336797</td>
      <td>21835.880400</td>
      <td>...</td>
      <td>0.381873</td>
      <td>0.0</td>
      <td>0.329446</td>
      <td>0.329446</td>
      <td>0.486161</td>
      <td>0.486161</td>
      <td>0.346637</td>
      <td>0.346637</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
      <td>0.060000</td>
      <td>30.890000</td>
      <td>4.000000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9000.000000</td>
      <td>0.088100</td>
      <td>265.730000</td>
      <td>5.000000e+04</td>
      <td>13.890000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>6293.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>15000.000000</td>
      <td>0.118000</td>
      <td>404.560000</td>
      <td>7.300000e+04</td>
      <td>19.760000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>12068.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>24000.000000</td>
      <td>0.155700</td>
      <td>648.100000</td>
      <td>1.040000e+05</td>
      <td>26.660000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>21735.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>40000.000000</td>
      <td>0.308400</td>
      <td>1676.230000</td>
      <td>8.797500e+06</td>
      <td>999.000000</td>
      <td>18.000000</td>
      <td>5.000000</td>
      <td>72.000000</td>
      <td>4.000000</td>
      <td>587191.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 95 columns</p>
</div>




```python
# Check the balance of our target values
y.value_counts()
```




    low_risk     68470
    high_risk      347
    Name: loan_status, dtype: int64




```python
# Split the X and y into X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=1)
```

## Data Pre-Processing

Scale the training and testing data using the `StandardScaler` from `sklearn`. Remember that when scaling the data, you only scale the features data (`X_train` and `X_testing`).


```python
# Create the StandardScaler instance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```


```python
# Fit the Standard Scaler with the training data
# When fitting scaling functions, only train on the training dataset
X_scaler = scaler.fit(X_train)
```


```python
# Scale the training and testing data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

# Ensemble Learners

In this section, you will compare two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble classifier . For each algorithm, be sure to complete the folliowing steps:

1. Train the model using the training data. 
2. Calculate the balanced accuracy score from sklearn.metrics.
3. Print the confusion matrix from sklearn.metrics.
4. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
5. For the Balanced Random Forest Classifier onely, print the feature importance sorted in descending order (most important feature to least important) along with the feature score

Note: Use a random state of 1 for each algorithm to ensure consistency between tests

### Balanced Random Forest Classifier


```python
# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

rf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
rf_model = rf_model.fit(X_train_scaled, y_train)

rf_model
```




    BalancedRandomForestClassifier(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = rf_model.predict(X_test_scaled)

balanced_accuracy_score(y_test, y_pred)
```




    0.7887512850910909




```python
# Display the confusion matrix

confusion_matrix(y_test, y_pred)
```




    array([[   71,    30],
           [ 2146, 14958]], dtype=int64)




```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.03      0.70      0.87      0.06      0.78      0.60       101
       low_risk       1.00      0.87      0.70      0.93      0.78      0.63     17104
    
    avg / total       0.99      0.87      0.70      0.93      0.78      0.63     17205
    
    


```python
# List the features sorted in descending order by feature importance
importances = rf_model.feature_importances_

for x,y in sorted(zip(rf_model.feature_importances_, X.columns), reverse=True):
    print(f"{y}: ({x})")
```

    total_rec_prncp: (0.07876809003486353)
    total_pymnt: (0.05883806887524815)
    total_pymnt_inv: (0.05625613759225244)
    total_rec_int: (0.05355513093134745)
    last_pymnt_amnt: (0.0500331813446525)
    int_rate: (0.02966959508700077)
    issue_d_Jan-2019: (0.021129125328012987)
    installment: (0.01980242888931366)
    dti: (0.01747062730041245)
    out_prncp_inv: (0.016858293184471483)
    total_bal_ex_mort: (0.01641297102011915)
    mths_since_recent_inq: (0.015220714904737209)
    issue_d_Mar-2019: (0.015115240704562424)
    revol_bal: (0.014926655663448373)
    max_bal_bc: (0.014899352873994727)
    out_prncp: (0.014881069023035237)
    mo_sin_old_rev_tl_op: (0.014859446582326507)
    annual_inc: (0.014832564501144122)
    bc_util: (0.014613819728800227)
    tot_hi_cred_lim: (0.014487685026878092)
    mo_sin_old_il_acct: (0.013921085423763812)
    mths_since_rcnt_il: (0.013534131593418711)
    total_rev_hi_lim: (0.013364759441576994)
    avg_cur_bal: (0.01332289882475225)
    bc_open_to_buy: (0.013265926832893358)
    il_util: (0.01311545089813887)
    total_bc_limit: (0.01304530062898567)
    all_util: (0.0130446065288952)
    pct_tl_nvr_dlq: (0.013030046723135838)
    tot_cur_bal: (0.012855901280381887)
    total_il_high_credit_limit: (0.01279908506759016)
    inq_last_12m: (0.012773576514405109)
    total_bal_il: (0.011968994260747247)
    mo_sin_rcnt_tl: (0.010982948025240226)
    loan_amnt: (0.010579906006851516)
    mths_since_recent_bc: (0.010575363106694519)
    num_rev_accts: (0.010320067009550682)
    total_acc: (0.010209212170253059)
    open_acc_6m: (0.009753839399393215)
    num_il_tl: (0.009662050208879065)
    num_actv_rev_tl: (0.009632472481996241)
    mo_sin_rcnt_rev_tl_op: (0.009393346012674945)
    num_bc_tl: (0.00872448189550355)
    num_sats: (0.008628938824946404)
    num_op_rev_tl: (0.008330966254402506)
    num_bc_sats: (0.007570544824579072)
    num_tl_op_past_12m: (0.007548811505974241)
    open_act_il: (0.007489717491934961)
    open_acc: (0.007382231721841728)
    num_rev_tl_bal_gt_0: (0.007272665006598051)
    total_rec_late_fee: (0.006998827313196186)
    num_actv_bc_tl: (0.006866662924995743)
    acc_open_past_24mths: (0.006714495620628373)
    open_il_24m: (0.006561432872333855)
    inq_fi: (0.006240598451492287)
    inq_last_6mths: (0.006216409633238659)
    mort_acc: (0.0061708920490257954)
    total_cu_tl: (0.006083218608279307)
    issue_d_Feb-2019: (0.005640206440873574)
    open_rv_24m: (0.005634546230136711)
    percent_bc_gt_75: (0.005131046989565006)
    next_pymnt_d_Apr-2019: (0.005106000423451099)
    next_pymnt_d_May-2019: (0.005036652777545191)
    open_il_12m: (0.004860024796675963)
    tot_coll_amt: (0.004198582835532627)
    open_rv_12m: (0.004018916067963884)
    num_accts_ever_120_pd: (0.0037571920083085985)
    delinq_2yrs: (0.003082852259926947)
    home_ownership_RENT: (0.0029133221443170495)
    home_ownership_OWN: (0.002824523629114469)
    verification_status_Not Verified: (0.002204946377565813)
    application_type_Joint App: (0.0020912385738361574)
    pub_rec: (0.002015258269512615)
    verification_status_Verified: (0.0019325773153555006)
    application_type_Individual: (0.001901604006185586)
    verification_status_Source Verified: (0.0015046400907840708)
    home_ownership_MORTGAGE: (0.0014589723334940362)
    pub_rec_bankruptcies: (0.0013727925120781853)
    initial_list_status_w: (0.0011520703643731528)
    initial_list_status_f: (0.0011005704165634263)
    num_tl_90g_dpd_24m: (0.0009956935704327383)
    collections_12_mths_ex_med: (0.0007150315534652695)
    home_ownership_ANY: (0.0004955956183545533)
    chargeoff_within_12_mths: (0.0002730803587770788)
    tax_liens: (0.0)
    recoveries: (0.0)
    pymnt_plan_n: (0.0)
    policy_code: (0.0)
    num_tl_30dpd: (0.0)
    num_tl_120dpd_2m: (0.0)
    hardship_flag_N: (0.0)
    delinq_amnt: (0.0)
    debt_settlement_flag_N: (0.0)
    collection_recovery_fee: (0.0)
    acc_now_delinq: (0.0)
    

### Easy Ensemble Classifier


```python
# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier

ee_model= EasyEnsembleClassifier(n_estimators= 100, random_state=1)

ee_model.fit(X_train_scaled, y_train)
```




    EasyEnsembleClassifier(n_estimators=100, random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = ee_model.predict(X_test_scaled)

balanced_accuracy_score(y_test, y_pred)
```




    0.931601605553446




```python
# Display the confusion matrix

confusion_matrix(y_test, y_pred)
```




    array([[   93,     8],
           [  985, 16119]], dtype=int64)




```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.09      0.92      0.94      0.16      0.93      0.87       101
       low_risk       1.00      0.94      0.92      0.97      0.93      0.87     17104
    
    avg / total       0.99      0.94      0.92      0.97      0.93      0.87     17205
    
    

# Which model had the best balanced accuracy score?
Easy Ensemble Classifier model has the best balanced accuracy score of 93.16%

# Which model had the best recall score?
Easy Ensemble Classifier model has the best recall score of 94%

# Which model had the best geometric mean score?
Easy Ensemble Classifier model has the best geometric mean score of 94%

# What are the top three features?
total_rec_prncp: (0.07876809003486353)

total_pymnt: (0.05883806887524815)

total_pymnt_inv: (0.05625613759225244)


```python

```
