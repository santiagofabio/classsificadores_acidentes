def analises_paramentrica(df_classifier,modelos_disribuicao_normal):
    
     #Teste ANOVA 
     # H0 =  Hipotese nula - os resultados são estatisticamente IGUAIS 
     # H1 = Hipotese alterantiva - os resultados são estatisticamente Diferentes
     import scipy.stats  as stats 
     statistic, pvalue = stats.f_oneway(df_classifier[modelos_disribuicao_normal[0]], df_classifier[modelos_disribuicao_normal[1]],
                                   df_classifier[modelos_disribuicao_normal[2]], df_classifier[modelos_disribuicao_normal[3]],
                                   df_classifier[modelos_disribuicao_normal[4]] )

     alpha =0.05
     print('-----------Teste ANOVA------------------')
     print(f'Estatistic, Value_P  {statistic} {pvalue }')
     if pvalue<= alpha:
             print('H0 rejeitatda ,H1 aceita.  Portanto, estatisticamente os dados são DIFERENTES')
             print('Pode ser aplicado o teste de Tukey')
     else:
             print('H0 aceita ,H1  rejeitada.  Portanto, estatisticamente os dados são IGUAIS')

     from scipy.stats import tukey_hsd 
     from statsmodels.stats.multicomp import MultiComparison

 

     lista_resultados =[]

     for i in range(0,5):
          lista_resultados.append(df_classifier[modelos_disribuicao_normal[i]])
     
     
     num_resultados = len(df_classifier[modelos_disribuicao_normal[i]])     


     lista_algoritmo = []
     for metodo in range(0,5):
         for experimento in range(0,num_resultados):
             lista_algoritmo.append(modelos_disribuicao_normal[metodo])
             
          
     import  numpy as np     
     resultados_algoritmos = {'accuracy': np.concatenate(lista_resultados),
                         'algoritmo': lista_algoritmo} 





     import pandas as pd 
     
     resultados_df = pd.DataFrame(resultados_algoritmos)
     compara_algoritmos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo']) 
     teste_estatistico = compara_algoritmos.tukeyhsd()
     print(f'{teste_estatistico}')

     
     import matplotlib.pyplot as plt
     fig2, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
     ax.set_xlabel("Estimators")
     plt.tight_layout()
     teste_estatistico.plot_simultaneous().savefig('Teste_tukey.jpg',dpi =400, format='jpg' )
     plt.close()         


    
     return(0)