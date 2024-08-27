import pandas as pd
from src.models.built_models import eval_model
import joblib
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_main():
    data = pd.read_csv('./files/datasets/intermediate/df_test.csv')
    features= data.drop(['is_ultra'],axis=1)
    target=data['is_ultra']
    seed=12345
    
    # Codificamos las variables categoricas
    rf=joblib.load('./files/modeling_output/model_fit/best_random_Random_Forest.joblib')
    dt=joblib.load('./files/modeling_output/model_fit/best_random_dt.joblib')
    lr=joblib.load('./files/modeling_output/model_fit/best_random_Logreg.joblib')
    models=[rf,dt,lr]
    models_name=['rf','dt','lr']
    test_results=[]
    for model,name in zip(models,models_name):
        prediction = model.predict(features)
        acc_val,f1_val,roc_auc_val = eval_model(model,features,target)
        test_results.append([name,acc_val,f1_val,roc_auc_val])
        cm = confusion_matrix(target,prediction)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
        fig, ax = plt.subplots()
        # Dibujar la matriz de confusión en el eje
        cm_display.plot(ax=ax)
        ax.set_title('Confusion_matrix')
        fig.savefig(f'./src/test/files/figs/fig_up_{name}.png')
    results_test= pd.DataFrame(test_results, columns=['model','acc_val','f1_val','roc_auc_val'])
    results_test.to_csv('./src/test/files/reports/test_report.csv',index=False)
    return results_test

results=test_main()

print(results)