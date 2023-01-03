def validacao_cruzada_knn(previsores,classe_alvo,num_teste_kfold):
     from sklearn.model_selection import KFold
     from sklearn.model_selection import cross_validate, cross_val_score
     from sklearn.neighbors import KNeighborsClassifier
     import matplotlib.pyplot as plt
     print('oi')
     resultados_modelo='k_nearest_neighbor'
     resultados_medios_knn =[]
     for i in range(0,num_teste_kfold):
            kfold = KFold(n_splits =10, shuffle =True, random_state =i)
            modelo =KNeighborsClassifier(n_neighbors =7, metric = "minkowski")
            resultado =cross_val_score(modelo,previsores,classe_alvo, cv =kfold)
            resultados_medios_knn.append(resultado.mean())
    
     
     
     import matplotlib.pyplot  as plt 
     import pandas as pd
     import seaborn as sns
     dataframe = pd.DataFrame({resultados_modelo:resultados_medios_knn})
     print(dataframe[resultados_modelo].describe())
     sns.kdeplot(data = dataframe, x=resultados_modelo,label='k_nearest_neighbor' )
     plt.legend(loc ='best')
     plt.tight_layout()
     plt.title('Classifier Distribution')
     plt.savefig('distribution_knn.jpg', dpi =300, format = 'jpg')
     plt.show()    
           

     return(dataframe)

