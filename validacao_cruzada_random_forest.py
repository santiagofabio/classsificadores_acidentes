def validacao_cruzada_random_forest(previsores,classe_alvo,num_teste_kfold):
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.model_selection import KFold
      from sklearn.model_selection import cross_validate, cross_val_score
      import time
      print('Inicio Validacao')
      inicio = time.time()
      resultados_medios_random_forest =[]
      random_forest =  RandomForestClassifier(n_estimators= 150,criterion = "gini", max_depth=4)
      for i  in range(0,num_teste_kfold):
             print(f'{i}')
             kfold = KFold(n_splits =10, shuffle =True, random_state =i)
             resultado =cross_val_score(random_forest,previsores,classe_alvo, cv =kfold)
             resultados_medios_random_forest.append(resultado.mean())
      print('FIM ')
      fim = time.time()
      tempo = fim - inicio
      with open('metrica_tempo_random.txt','w') as f : 
                   f.write('Tempo de execussao random: {:.3}\n'.format(tempo))

      import matplotlib.pyplot  as plt 
      import pandas as pd
      import seaborn as sns
      nome_modelo ='RandomForestClassifier'
      dataframe = pd.DataFrame({nome_modelo:resultados_medios_random_forest})
      dataframe.to_csv('validacao_randomforest.csv',sep=';', encoding='utf-8', index =False)
      print(dataframe[nome_modelo].describe())
      sns.kdeplot(data = dataframe, x=nome_modelo, label ="RandomForestClassifier" )
      plt.tight_layout()
      plt.legend(loc ='best')
      plt.title('Classifier distribution')
      plt.savefig('distribution_random_forest_classifier.jpg', dpi =300, format = 'jpg')
      plt.show() 
     
     
      return(dataframe)