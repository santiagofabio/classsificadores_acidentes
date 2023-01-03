def validacao_cruzada_lgbm(previsores,classe_alvo,num_teste_kfold):
      from sklearn.model_selection import KFold
      from sklearn.model_selection import cross_val_score
      import lightgbm as lgb 
      import time
      nome_modelo ='lightgbm'
      resultados_medios_lightgbm=[]
      inicio = time.time()
      for i in range(0,num_teste_kfold):
            kfold = KFold(n_splits=10, shuffle=True, random_state=i)
            modelo = lgb.LGBMClassifier( num_leaves=250, objective='binary',
                                  max_depth=2, learning_rate=0.05, max_bin =100)
            resultado = cross_val_score(modelo, previsores, classe_alvo, cv=kfold)
            resultados_medios_lightgbm.append(resultado.mean())
      
      fim = time.time()
      tempo = fim - inicio
      with open('metrica_tempo_lgbm.txt','w') as f : 
                   f.write('Tempo de execussao lgbm: {:.3}\n'.format(tempo))

      
      import matplotlib.pyplot  as plt 
      import pandas as pd
      import seaborn as sns
      dataframe = pd.DataFrame({nome_modelo:resultados_medios_lightgbm})
      dataframe.to_csv('validacao_lightgbm.csv',sep=';', encoding='utf-8', index=False)
      print(dataframe[nome_modelo].describe())
      sns.kdeplot(data = dataframe, x=nome_modelo, label =nome_modelo  )
      plt.legend(loc ='best')
      plt.tight_layout()
      plt.title('Classifier distribution')
      plt.savefig('distribution_lightgbm.jpg', dpi =300, format = 'jpg')
      plt.show()    
           
      
      return(dataframe)