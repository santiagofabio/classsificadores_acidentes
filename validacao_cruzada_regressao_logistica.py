def validacao_cruzada_regressao_logistica(previsores,classe_alvo,num_teste_kfold ):
      from sklearn.model_selection import KFold
      from sklearn.model_selection import cross_validate, cross_val_score
      from sklearn.linear_model import LogisticRegression
      import matplotlib.pyplot as plt
      import time
      inicio = time.time()
      
      resultado_logistica =[]
      for i  in range (0,num_teste_kfold):
             kfold = KFold(n_splits =10, shuffle =True, random_state =i)
             modelo =LogisticRegression(random_state=i, max_iter=300, penalty="l2")
             resultado =cross_val_score(modelo,previsores,classe_alvo, cv =kfold)
             resultado_logistica.append(resultado.mean())
       
      fim = time.time()
      tempo = fim - inicio
      with open('metrica_tempo_logistica.txt','w') as f : 
                   f.write('Tempo de execussao logistica: {:.3}\n'.format(tempo))
      
      import matplotlib.pyplot  as plt 
      import pandas as pd
      import seaborn as sns
      resultados_modelo ='regressao_logistica'
      dataframe = pd.DataFrame({resultados_modelo:resultado_logistica})
      dataframe.to_csv('validacao_regressao_logistica.csv',sep=';', encoding='utf-8', index =False)

      print(dataframe[resultados_modelo].describe())
      sns.kdeplot(data = dataframe, x=resultados_modelo,label='Distribution Logistica')
      plt.tight_layout()
      plt.legend(loc ='best')
      plt.title('Estimador')
      plt.savefig('distribution_logistica.jpg', dpi =300, format = 'jpg')
      plt.show()    
      
      

      return(dataframe)