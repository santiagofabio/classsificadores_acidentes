def validacao_cruzada_svm(previsores,classe_alvo,num_teste_kfold):
      from sklearn.model_selection import KFold
      from sklearn.model_selection import cross_validate, cross_val_score
      from sklearn.svm import SVC
      import matplotlib.pyplot as plt
      import time 
      resultados_modelo = 'resultados_svc'
      
      inicio = time.time()
      resultados_svc = []
      for i in range(0,num_teste_kfold):
              kfold = KFold(n_splits =10, shuffle =True, random_state =i)
              modelo =SVC()
              resultado =cross_val_score(modelo,previsores,classe_alvo, cv =kfold)
              media = resultado.mean()
              resultados_svc.append(media )
      fim = time.time()
      tempo = fim - inicio
      with open('metrica_tempo_svm.txt','w') as f : 
                   f.write('Tempo de execussao svm: {:.3}\n'.format(tempo))
 
      
      import matplotlib.pyplot  as plt 
      import pandas as pd
      import seaborn as sns
      dataframe = pd.DataFrame({resultados_modelo:resultados_svc})
      dataframe.to_csv('validacao_svc.csv',sep=';', encoding='utf-8', index =False)
      print(dataframe[resultados_modelo].describe())
      sns.kdeplot(data = dataframe, x=resultados_modelo,label='Distribution SVM')
      plt.legend(loc ='best')
      plt.tight_layout()
      plt.title('Classifier distribution')
      plt.savefig('distribution_svc.jpg', dpi =300, format = 'jpg')
          
      plt.show()    
      return(dataframe)