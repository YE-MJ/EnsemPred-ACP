import os
import pandas as pd
import joblib
import numpy as np

def RF_ensemble(df1, df2, df3, df4, df5, df6, df7, name1, name2, name3, name4, name5, name6, name7):
    dataframes = [df1, df2, df3, df4, df5, df6, df7]
    names = [name1, name2, name3, name4, name5, name6, name7]

    all_predictions = []

    for df, name in zip(dataframes, names):
        X_test = df.drop(['SampleName'], axis=1)
        
        tmp = [] 
        
        weight_file_paths = [
                f"./selected/best_model_RF_{name}_fold_1.pkl",
                f"./selected/best_model_RF_{name}_fold_2.pkl",
                f"./selected/best_model_RF_{name}_fold_3.pkl",
                f"./selected/best_model_RF_{name}_fold_4.pkl",
                f"./selected/best_model_RF_{name}_fold_5.pkl"
        ]

        for weight_file_path in weight_file_paths:
            model = joblib.load(weight_file_path)
            y_pred = model.predict(X_test)
            tmp.append(y_pred) 

        y_preds_ensemble = np.mean(tmp, axis=0)
        
        all_predictions.append(y_preds_ensemble)
    
    final_prediction = np.mean(all_predictions, axis=0)
    
    return final_prediction

