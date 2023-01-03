def validacao_cruzada_knn(previsores,classe_alvo,num_teste_kfold):
       from sklearn.model_selection import KFold
       from sklearn.model_selection import cross_validate, cross_val_score
       from sklearn.neighbors import KNeighborsClassifier
       import matplotlib.pyplot as plt
       import time
       nome_modelo ='knearestneighbor'
       inicio = time.time()
       resultados_medios_knn =[]
     
       for i in range(0,num_teste_kfold):
             kfold = KFold(n_splits =10, shuffle =True, random_state =i)
             modelo =KNeighborsClassifier()
             resultado =cross_val_score(modelo,previsores,classe_alvo, cv =kfold)
             resultados_medios_knn.append(resultado.mean())
     
       fim = time.time()
       tempo = fim - inicio
       with open('metrica_tempo_knn.txt','w') as f : 
                   f.write('Tempo de execussao knn: {:.3}\n'.format(tempo))
              
       import matplotlib.pyplot  as plt 
       import pandas as pd
       import seaborn as sns
       from matplotlib.pyplot import rcParams
       rcParams["figure.figsize"] = (7,7)
     
       dataframe = pd.DataFrame({nome_modelo:resultados_medios_knn})
       dataframe.to_csv('validacao_knn.csv',sep=';', encoding='utf-8', index =False)
       sns.kdeplot(data = dataframe,x='knearestneighbor', label='Distribution KNN')
       plt.legend(loc ='best')
       plt.tight_layout()
       plt.title('Classifier distribution')
       plt.savefig('distribution_knn.jpg', dpi =300, format = 'jpg')
       plt.show()    
     

       return(dataframe)

