# %% [markdown]
# # Descripción del proyecto  
# 
# La compañía móvil Megaline no está satisfecha al ver que muchos de sus clientes utilizan planes heredados. Quieren desarrollar un modelo que pueda analizar el comportamiento de los clientes y recomendar uno de los nuevos planes de Megaline: Smart o Ultra.
# Tienes acceso a los datos de comportamiento de los suscriptores que ya se han cambiado a los planes nuevos (del proyecto del curso de Análisis estadístico de datos). Para esta tarea de clasificación debes crear un modelo que escoja el plan correcto. Como ya hiciste el paso de procesar los datos, puedes lanzarte directo a crear el modelo.
# Desarrolla un modelo con la mayor exactitud posible. En este proyecto, el umbral de exactitud es 0.75. Usa el dataset para comprobar la exactitud.
# 
# ## Instrucciones del proyecto.
# 1. Abre y examina el archivo de datos. Dirección al archivo:datasets/users_behavior.csv 
# 2. Segmenta los datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba.
# 3. Investiga la calidad de diferentes modelos cambiando los hiperparámetros. Describe brevemente los hallazgos del estudio.
# 4. Comprueba la calidad del modelo usando el conjunto de prueba.
# 5. Tarea adicional: haz una prueba de cordura al modelo. Estos datos son más complejos que los que habías usado antes así que no será una tarea fácil. Más adelante lo veremos con más detalle.
# 
# ## Descripción de datos
# Cada observación en el dataset contiene información del comportamiento mensual sobre un usuario. La información dada es la siguiente:
# - сalls — número de llamadas,
# - minutes — duración total de la llamada en minutos,
# - messages — número de mensajes de texto,
# - mb_used — Tráfico de Internet utilizado en MB,
# - is_ultra — plan para el mes actual (Ultra - 1, Smart - 0).

# %% [markdown]
# ## Cargar librerías

# %%
#Cargar todas las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from pprint import pprint
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# %% [markdown]
# ## Cargar datos

# %%
df=pd.read_csv('datasets/users_behavior.csv')
df.head(10)

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# Haciendo una visualización inicial de los datos, tenemos 5 columnas y 3214 filas, de los cuales a simple vista no vemos ningún ausente. Ya veremos más a fondo esto. 

# %%
#Calculamos los ausentes
print('Ausentes:\n',df.isna().sum())

# %%
#Calculamos los duplicados
print('Duplicados:\n',df.duplicated().sum())


# %% [markdown]
# Podemos evidenciar que no contamos con valores ausentes ni valores duplicados en el dataset y tenemos datos limpios listos para el modelo de clasificación.

# %% [markdown]
# ## Segmentación de los datos

# %%
#Usamos la semilla para los datos pseudoaleatorios
seed=54321
#Segmentamos primero los datos de entrenamiento y los datos de prueba
df_train,df_test=train_test_split(df,test_size=0.2,random_state=seed)
features=df_train.drop('is_ultra',axis=1)
target=df_train['is_ultra']

# %%
#Segmentamos ahora los datos de entrenamiento y validación
features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.2,random_state=seed)

# %% [markdown]
# Ahora tenemos 80% de datos de entrenamiento, 20% de datos de validación y 20% de datos de prueba. A continuación vamos a entrenar los modelos.

# %% [markdown]
# ## Prueba de modelos de clasificación

# %% [markdown]
# ### Arbol de desición

# %%
best_score_1=0
best_depth_1=0
for depth in range(1,11):
    model_1=DecisionTreeClassifier(random_state=seed,max_depth=depth)
    model_1.fit(features_train,target_train)
    prediction_1=model_1.predict(features_valid)
    result_1=metrics.accuracy_score(target_valid,prediction_1)
    if result_1 > best_score_1:
        best_score_1=result_1
        best_depth_1=depth
print(f'max_depth = {depth} : {result_1}')


# %% [markdown]
# Obtenemos una exactitud del 78% en el arbol de desición con una profundidad de 10 ramas en el arbol, sin embargo, es una profundidad muy alta que puede costarnos un sobreentrenamiento.

# %% [markdown]
# ### Bosque Aleatorio

# %%
best_depth_2=0
best_estimator_2=0
best_score_2=0
for est in range(1,51):
    for depth in range(1,10):
        model_2=RandomForestClassifier(random_state=seed,max_depth=depth,n_estimators=est)
        model_2.fit(features_train,target_train)
        prediction_2 = model_2.predict(features_valid)
        result_2=metrics.accuracy_score(target_valid,prediction_2)
        if result_2 > best_score_2:
            best_score_2=result_2
            best_estimator_2=est
            best_depth_2=depth
print(f'El mejor resultado es {best_score_2} con {best_estimator_2} estimadores y {best_depth_2} de profundidad.')


# %% [markdown]
# El mejor resultado hasta el momento es el bosque aleatorio con 11 estimadores y 4 de profundidad del arbol. Al tener varios estimadores, no nos preocupa el sobreentrenamiento, además la exactitud es de 82%.

# %% [markdown]
# ### Regresión Logística

# %%
model_3=LogisticRegression(random_state=seed,solver='liblinear')
model_3.fit(features_train,target_train)
result_3_train=model_3.score(features_train,target_train)
result_3_valid=model_3.score(features_valid,target_valid)
print(f'Resultado entrenamiento: {result_3_train}\nResultado validación: {result_3_valid}')

# %% [markdown]
# La regresión logística es un algoritmo rápido, sin embargo, tiene el porcentaje de exactitud más bajo con 71,8%.

# %% [markdown]
# El modelo escogido para la prueba con el conjunto de prueba es el **bosque aleatorio con 4 de profundidad y 11 estimadores.**

# %% [markdown]
# ### Evaluación del modelo

# %%
features_test=df_test.drop('is_ultra',axis=1)
target_test=df_test['is_ultra']

# %%
model_final=RandomForestClassifier(random_state=seed,max_depth=4,n_estimators=11)
model_final.fit(features_train,target_train)
prediction_test=model_final.predict(features_test)
result_final=metrics.accuracy_score(target_test,prediction_test)
print(f'El resultado final tiene una exactitud de {result_final*100}%.')

# %%
print(metrics.classification_report(target_test, prediction_test))

# %% [markdown]
# Al evaluar el modelo obtenemos una exactitud del 78,53%, mayor a lo esperado que era 75%, por lo cual podemos concluír que el algoritmo es adecuado para la predicción del dataset de prueba.
# 
# Adicionalmente al ver las otras métricas nuestra precisión es buena para ambos datasets, sin embargo, al estar nuestros datos desbalanceados, el recall de la clase 1 es mas baja.

# %% [markdown]
# ### Prueba de cordura

# %%
#Para hacer la prueba de cordura, vamos a comprar el modelo escogido con el modelo DummyClassifier 
model_safe_test=DummyClassifier(random_state=seed)
model_safe_test.fit(features_train,target_train)
prediction_safe_t=model_safe_test.predict(features_test)
result_safe_t=metrics.accuracy_score(target_test,prediction_safe_t)
print(f'El resultado final tiene una exactitud de {result_safe_t*100}%.')
print(f'La diferencia es del {(result_final-result_safe_t)*100}%')

# %% [markdown]
# Al realizar la prueba de cordura, podemos ver que la calidad del modelo es más alta (**78%**) que la del modelo DummyClassifier (**Exactitud: 66%**), por lo cual nuestro modelo pasa la prueba.

# %% [markdown]
# ### Mejora del modelo con RandomizedCV

# %%
#Primero revisamos los parametros que se están usando actualmente
model_to_improve=RandomForestClassifier()
print('Parameters currently in use:\n')
pprint(model_to_improve.get_params())

# %%
#Ahora vamos a pasar la lista de parametros que queremos iterar:
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Número de features a considerar para cada separación
max_features = randint(1, 11)
# Máximo número de niveles a considerar en el arbol
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Número mínimo  de pruebas requeridas para las eparación de un nodo
min_samples_split = [2, 5, 10]
# Número minimo de pruebas requeridas para cada nodo hoja
min_samples_leaf = [1, 2, 4]
# Metodo de selección de pruebas para el entrenamiento de cada árbol
bootstrap = [True, False]
# Creación de la malla aleatoria
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# %%
#Ahora junto con la malla y el RandomizedCV vamos a generar el mejor modelo con los mejores hiperparametros 
model_random = RandomizedSearchCV(estimator = model_to_improve, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=seed, n_jobs = -1)
# Entrenamos el modelo
model_random.fit(features_train,target_train)
print(model_random.best_params_)

# %%
best_random = model_random.best_estimator_
random_prediction = best_random.predict(features_test)
random_accuracy=metrics.accuracy_score(target_test,random_prediction)
print("Accuracy:",random_accuracy)

# %% [markdown]
# Con este método, logramos mejorar la exactitud del modelo a 79%

# %% [markdown]
# ## Conclusión
# 
# El mejor modelo de clasificación es el **Bosque Aleatorio** con un **82%** de exactitud con el dataset de validación y **78%** con el modelo de prueba.
# 
# Si análizamos la exactitud del modelo, podemos ver que el modelo adivinó el 81% de los usuarios con plan Ultra y el 78% de los usuarios con plan Smart.
# 
# Además, tenemos un recall de 95% de los de plan ultra y 47% de plan smart, lo que significa que predijo más cantidad de personas con el plan ultra y menos con el plan smart, por lo que tenemos una oportunidad de mejora frente a esto.
# 
# Finalmente al hacer la prueba de cordura, nuestro modelo pasa la prueba con una diferencia del 12% por encima, lo que indica que nuestro modelo es bueno.


