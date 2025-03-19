# Full result table 

We provide the fine-grained results of all the experiments presented in the paper in the attached `full_results.csv` file. 
Each row of the csv file is the result of one particular model, for one particular species and phenophase, on one particular fold, and with one particular data splitting stratgy. 

The csv file has the following columns :
- model_id: Name of the model 
- kind: Type of model (DL: deep learning, ML: traditional machine learning, PM: process-based model)
- fold: fold number (out of the 40 folds)
- split_mode: data splitting strategy (random, random spatial, random temporal, structured temporal)
- species: species name 
- phase: phenophase name
- target_id: species+phenophase
- r2: R2 score
- rmse: Root mean squared error
- mae: Mean absolute error
- mape: Mean absolute percentage error
- nrmse: Normalised RMSE (RMSE divided by the variance of the true values)