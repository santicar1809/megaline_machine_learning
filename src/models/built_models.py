import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay
from src.models.hyper_parameters import all_models
import joblib

def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''
    
    models = all_models() 
    
    output_path = './files/modeling_output/model_fit/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    results = []

    # Iterating the models
    models_name = ['rf','dt','lr','dummie']
    for model in models:
        best_estimator, best_score, acc_val,f1_val,roc_auc_val= model_structure(data, model[1], model[2])[0]
        random_predict = model_structure(data, model[1], model[2])[1]
        target_valid = model_structure(data, model[1], model[2])[2]
        results.append([model[0],best_score, acc_val,f1_val,roc_auc_val])
        # Confusion matrix
        cm = confusion_matrix(target_valid,random_predict)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
        fig, ax = plt.subplots()
        cm_display.plot(ax=ax)
        ax.set_title('Confusion_matrix')
        fig.savefig(f'./files/modeling_output/figures/fig_{model[0]}')
        
        # Guardamos el modelo
        joblib.dump(best_estimator,output_path +f'best_random_{model[0]}.joblib')
    results_df = pd.DataFrame(results, columns=['model','best_train_score','acc_val','f1_val','roc_auc_val'])
    results_df.to_csv('./files/modeling_output/reports/model_report.csv',index=False)
    return results_df

def model_structure(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    #Usamos la semilla para los datos pseudoaleatorios
    seed=54321
    #Segmentamos primero los datos de entrenamiento y los datos de prueba
    df_train,df_test=train_test_split(data,test_size=0.2,random_state=seed)
    features=df_train.drop('is_ultra',axis=1)
    target=df_train['is_ultra']
    df_test.to_csv('./files/datasets/intermediate/df_test.csv',index=False)
    #Segmentamos ahora los datos de entrenamiento y validación
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.2,random_state=seed)
    
    #Gráficamos las frecuencias relativas de cada clase
    fig,ax=plt.subplots()
    balance=target_train.value_counts(normalize=True)
    ax.bar(balance.index.astype(str),balance)
    ax.set_title('Balance of clases')
    fig.savefig('./files/modeling_output/figures/fig_balance.png')
    
    # Training the model
    gs = RandomizedSearchCV(pipeline, param_grid, cv=2, scoring='f1', n_jobs=-1, verbose=2)
    gs.fit(features_train,target_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    best_prediction = best_estimator.predict(features_valid)
    acc_val,f1_val,roc_auc_val = eval_model(best_estimator,features_valid,target_valid)
    
    results = best_estimator, best_score,acc_val,f1_val,roc_auc_val
    return results, best_prediction,target_valid
    
def eval_model(best,features_valid,target_valid):
    random_prediction = best.predict(features_valid)
    random_prob=best.predict_proba(features_valid)[:, 1]
    accuracy_val=accuracy_score(target_valid,random_prediction)
    f1_val=f1_score(target_valid,random_prediction)
    roc_auc_val= roc_auc_score(target_valid,random_prob)
    return accuracy_val,f1_val,roc_auc_val