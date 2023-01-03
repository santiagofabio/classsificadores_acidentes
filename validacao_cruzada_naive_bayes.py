def  validacao_cruzada_naive_bayes(previsores,classe_alvo,num_teste_kfold):
         from sklearn.model_selection import KFold
         from sklearn.model_selection import cross_validate, cross_val_score
         from sklearn.naive_bayes import GaussianNB
         import time

         inicio = time.time()
         
         resultados_naive_bayes = []
      
         for i in range(0,num_teste_kfold):
              kfold = KFold(n_splits =10, shuffle =True, random_state =i)
              modelo =GaussianNB()
              resultado =cross_val_score(modelo,previsores,classe_alvo, cv =kfold)
              resultados_naive_bayes.append(resultado.mean())
        
         fim = time.time()
         tempo = fim - inicio
         with open('naive_bayes_tempo.txt','w') as f : 
                   f.write('Tempo de execussao naive bayes: {:.3}\n'.format(tempo))
        
         import matplotlib.pyplot  as plt 
         import pandas as pd
         import seaborn as sns
         nome_modelo ='Naive_Bayes'
         from matplotlib.pyplot import rcParams
         rcParams["figure.figsize"] = (7,7)
         dataframe = pd.DataFrame({nome_modelo:resultados_naive_bayes})
      
         dataframe.to_csv('validacao_naive_bayes.csv',sep=';', encoding='utf-8', index=False)
    
         sns.kdeplot(data = dataframe, x= nome_modelo,label='Distribution Naive Bayes')
         #sns.histplot(data = dataframe, x= nome_modelo,label='Distribution Naive Bayes', kde=True)
         plt.legend(loc ='best')
         plt.tight_layout()
         plt.title('Classifier distribution')
         plt.savefig('distribution_naive_bayes.jpg', dpi =300, format = 'jpg')
         plt.show()
 
    
    
         return( dataframe)