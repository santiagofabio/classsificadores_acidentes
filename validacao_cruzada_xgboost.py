def validacao_cruzada_xgboost(previsores,classe_alvo, num_teste_kfold):
      from xgboost import XGBClassifier
      from sklearn.model_selection import cross_validate, cross_val_score
      from sklearn.model_selection import KFold
   
      import matplotlib.pyplot as plt
      import time
      nome_modelo ='XGBClassifier'
      inicio = time.time()
      resultados_medios_xgb =[]
      for i  in range(0,num_teste_kfold):
            print(f'{i}')
            kfold = KFold(n_splits =10, shuffle =True, random_state =i)
            xgboost =XGBClassifier(max_depth =4, n_estimators =200)
            resultado =cross_val_score(xgboost,previsores,classe_alvo, cv =kfold)
            resultados_medios_xgb.append(resultado.mean())
      
      fim = time.time()
      tempo = fim - inicio
      with open('metrica_tempo_xgboost.txt','w') as f : 
                   f.write('Tempo de execussao xgboost: {:.3}\n'.format(tempo))
     
      import matplotlib.pyplot  as plt 
      import pandas as pd
      import seaborn as sns
      dataframe = pd.DataFrame({nome_modelo:resultados_medios_xgb})
      dataframe.to_csv('validacao_xgb.csv',sep=';', encoding='utf-8',index=False)

      print(dataframe[nome_modelo].describe())
      sns.kdeplot(data = dataframe, x=nome_modelo,label ='XGBClassifier')
      plt.legend(loc ='best')
      plt.tight_layout()
      plt.title('Classifier distribution')
      plt.savefig('distribution_xgb.jpg', dpi =300, format = 'jpg')
      plt.show()    
      return(dataframe)