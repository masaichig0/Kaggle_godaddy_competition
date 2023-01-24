# Kaggle_godaddy_competition
### The Process of Kaggle competition. 


When I get the result with tensorflow Dense model with 1.093, I started to think how I can improve the result, and this is the process:
* Notebook; multiprocess.ipynb
  Because tensorflow model takes long time to get the result, I decided to use multiprocess to speed up running time, and this is the code for it. 
  I trained first 31 period with each country, then test with last 8 periods. I evaluate and get MAPE. CSV file dense_results_full_data.csv is the reslt of this notebook. 
  
* Notebook; analyze_results.ipynb
  Analyze the results form tensorflow Dense model. I categorize 1 and 2 based on the MAPE results. I was intended to get forecast with Linear Regression model and evaluate
  it ot try increase total MAPE. I save the reuslt in CSV file result_dense_with_category.csv
  
* Notebook; Linear_regression_with_category2.ipynb
  Trained and get MAPE with linear regression the country labeled category 2. I used the same method of first notebook with linear regression. One of the bad SMAPE, I used
  the same value as last period, and used as a forecast. Save the result on CSV result_after_LR.csv
  
* Notebook; Check_MAPE_results.ipynb
  I decided to get MAPE for entire dataset with each train model (tf dense, LR, and same values), then compare the results. I categorized the best score for next step. 
  I saved the result on CSV file compare_mape.csv. 
  
* Notebook; train_multiple_models.ipynb
  Use the result of comapare_mape.csv and trained on multiple methods. The result improved significantly and I am ready to apply kaggle competition. I saved the result on CSV file forecast_with_category.csv 
  
* helper_functions.py
  All the functions I used in those notebook is in here.
  
### Next Step is to apply Kaggle competition on the multiple models traing method to get submission.  

Link: https://www.kaggle.com/code/masahiroichigo/multiple-model/edit
