from classificador_naive_bayes import classificador_naive_bayes
from validacao_cruzada_naive_bayes import validacao_cruzada_naive_bayes 
from validacao_cruzada_svm import validacao_cruzada_svm
from validacao_cruzada_knn import validacao_cruzada_knn
from validacao_cruzada_regressao_logistica import validacao_cruzada_regressao_logistica
from classificador_regressao_logistica import classificador_regressao_logistica
from  validacao_cruzada_arvores_decisao import  validacao_cruzada_arvores_decisao 
from  classificador_random_forest import classificador_random_forest
from  classificador_knn import classificador_knn 
from classificador_arvores_dedecisao import  classificador_arvores_dedecisao
from validacao_cruzada_random_forest import validacao_cruzada_random_forest 
from classficador_svm import classficador_svm 
from classificador_lgbm import classificador_lgbm
from validacao_cruzada_xgboost import validacao_cruzada_xgboost 
from validacao_cruzada_lgbm import validacao_cruzada_lgbm 
from classificador_xgboost import classificador_xgboost 
import pandas as pd
import numpy as np
from exploracao_dados import exploracao_dados 
import seaborn as sns 
import matplotlib.pyplot as plt
from preprocessamento_dados2 import preprocessamento_dados2 
import pickle


file = 'dataset.csv'
dataset = pd.read_csv(file, sep =';',  encoding= 'utf-8')
print(dataset.columns)
print(dataset.shape)
print(dataset.isnull().sum())
print(dataset.dtypes)

"""
Catalogo de dados 
fumante                object
bebidas                object
drogas                 object
acidentes             float64
trabalho excessivo     object
posicao incomoda       object
horas dormindas       float64
fss                   float64
fadiga                float64
horas trabalhadas     float64
ess                   float64
"""




dataset.dropna(inplace=True, axis=0)

print(dataset.columns)
print(dataset.shape)
print(dataset.dtypes)


print(dataset.shape)
preprocessamento_dados2(dataset)
from exploracao_quantitativo import exploracao_quantitativo 
exploracao_quantitativo(dataset) 
from exploracao_dados import exploracao_dados
exploracao_dados(dataset)


with open('previsores_escalonados.pkl','rb') as arquivo:
             previsores_escalodados = pickle.load(arquivo)
     
with open('classe_alvo.pkl','rb') as arquivo:
             classe_alvo=pickle.load(arquivo)

print(previsores_escalodados.shape)
print(classe_alvo.shape)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino,y_teste =train_test_split( previsores_escalodados,classe_alvo, test_size =0.3,random_state=0)
print(f'x_treino.sphae: {x_treino.shape}')
print(f'y_treino.shape:{y_treino.shape}')
print(f'x_test.shape: {x_teste.shape}')
print(f'y_test.shape: {y_teste.shape}')

from vizualizacao_confusion_matrix import vizualizacao_confusion_matrix
vizualizacao_confusion_matrix(x_treino, x_teste, y_treino,y_teste)



from visualizacao_recall_precision import visualizacao_recall_precision
visualizacao_recall_precision(x_treino, x_teste, y_treino,y_teste,previsores_escalodados,classe_alvo )







# Validação K-FOLD

num_teste_kfold =40
classificador_naive_bayes(x_treino, x_teste, y_treino,y_teste)
df_naive_bayes= validacao_cruzada_naive_bayes(previsores_escalodados,classe_alvo,num_teste_kfold )

classificador_knn(x_treino, x_teste, y_treino,y_teste)
df_knn = validacao_cruzada_knn(previsores_escalodados,classe_alvo,num_teste_kfold )

classificador_arvores_dedecisao(x_treino, x_teste, y_treino,y_teste)
df_arvores = validacao_cruzada_arvores_decisao(previsores_escalodados,classe_alvo,num_teste_kfold)

classificador_regressao_logistica(x_treino, x_teste, y_treino,y_teste)
df_regressao_logistica=validacao_cruzada_regressao_logistica(previsores_escalodados,classe_alvo,num_teste_kfold )

classificador_random_forest(x_treino, x_teste, y_treino,y_teste)
df_random_forest = validacao_cruzada_random_forest(previsores_escalodados,classe_alvo,num_teste_kfold)

classficador_svm(x_treino, x_teste, y_treino,y_teste)
df_svm = validacao_cruzada_svm(previsores_escalodados,classe_alvo,num_teste_kfold)

classificador_lgbm(x_treino, x_teste, y_treino,y_teste)
df_lgbm =validacao_cruzada_lgbm(previsores_escalodados,classe_alvo,num_teste_kfold)

classificador_xgboost(x_treino, x_teste, y_treino,y_teste)
df_xgboost = validacao_cruzada_xgboost(previsores_escalodados,classe_alvo,num_teste_kfold)
#-----------------------------------------------------------------







df_random_forest = pd.read_csv('validacao_randomforest.csv', sep =';', encoding='utf-8')
df_xgboost = pd.read_csv("validacao_xgb.csv", sep =';', encoding='utf-8')
df_svm = pd.read_csv("validacao_svc.csv", sep =';', encoding='utf-8')
df_lgbm = pd.read_csv("validacao_lightgbm.csv", sep =';', encoding='utf-8')
df_knn = pd.read_csv("validacao_knn.csv", sep =';', encoding='utf-8')
df_arvores = pd.read_csv("validacao_arvores_decisao.csv", sep =';', encoding='utf-8')
df_regressao_logistica = pd.read_csv('validacao_regressao_logistica.csv', sep =';', encoding='utf-8')
df_naive_bayes = pd.read_csv('validacao_naive_bayes.csv', sep =';', encoding='utf-8')








file = 'df_classifier.csv'
df_classifier= pd.read_csv(file, sep =';', encoding='utf-8')     
#print(df_classifier)
label_modelos = ['Naive Bayes', 'KNN', 'Decision Tree',
       'Random Forest', 'SVM', 'Logistic Regression',
       'Lightgbm', 'XGBClassifier']


print(df_classifier.describe())




# Teste de Shapiro-Willk 
alpha =0.05
from scipy.stats import shapiro
modelos_nao_normal =[]
modelos_disribuicao_normal =[]
for modelo in label_modelos:
        statistic,p_value = shapiro(df_classifier[modelo])
        print('Modelo {}: {:.4f}'.format(modelo, p_value))
        if p_value<= alpha:
             modelos_nao_normal.append(modelo)
        else:
              modelos_disribuicao_normal.append(modelo)    

print('Modelos normalmente distribuidos')
print(modelos_disribuicao_normal)
print(len(modelos_disribuicao_normal))
print('Modelos não normal')
print(len(modelos_nao_normal) )







from visualizacao_kfold import visualizacao_kfold
visualizacao_kfold(df_classifier,modelos_disribuicao_normal,modelos_nao_normal)

from analises_parametricas import analises_parametricas
analises_parametricas(df_classifier,modelos_disribuicao_normal)

from analises_nao_parametricas import analises_nao_parametricas
analises_nao_parametricas(df_classifier,modelos_nao_normal)

