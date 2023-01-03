def validacao_cruzada_arvores_decisao(previsores,classe_alvo,num_teste_kfold):
    
      from sklearn.model_selection import KFold
      from sklearn.model_selection import cross_validate, cross_val_score
      import matplotlib.pyplot as plt
      from sklearn.tree import DecisionTreeClassifier
      import time
      inicio = time.time()
      
      resultados_medio_arvore =[]
      for i in range(0,num_teste_kfold):   
             kfold = KFold(n_splits =10, shuffle =True, random_state =i)
             arvore = DecisionTreeClassifier(criterion = "entropy", random_state=0 , max_depth=3)
             resultado =cross_val_score(arvore,previsores,classe_alvo, cv =kfold)
             resultados_medio_arvore.append(resultado.mean())

      fim = time.time()
      tempo = fim - inicio   
      with open('metrica_tempo_arvore.txt','w') as f : 
               f.write('Tempo de execussao arvore: {:.3}\n'.format(tempo))  
      
      import matplotlib.pyplot  as plt 
      import pandas as pd
      import seaborn as sns
      
      from matplotlib.pyplot import rcParams
      rcParams["figure.figsize"] = (7,7)
      
      nome_modelo ='DecisionTreeClassifier'
      dataframe = pd.DataFrame({nome_modelo:resultados_medio_arvore})
      dataframe.to_csv('validacao_arvores_decisao.csv',sep=';', encoding='utf-8', index=False)
      print(dataframe[nome_modelo].describe())
      sns.kdeplot(data = dataframe, x=nome_modelo, label =nome_modelo  )
      plt.legend(loc ='best')
      plt.tight_layout()
      plt.title('Classifier distribution')
      plt.savefig('distribution_decision_tree.jpg', dpi =300, format = 'jpg')
      plt.show()    
           
             
             
   
      return(dataframe )