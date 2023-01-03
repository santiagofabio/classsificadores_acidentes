def analises_nao_parametricas(df_classifier,modelos_nao_normal):
       from scipy import stats
    
       #Teste kruskal
       # H0 =  Hipotese nula  that the population median of all of the groups are equal
       # H1 =  Hipotese alterantiva - that the population median of all of the groups are diffrente

       print('-----------Teste Kruskal------------------')

       statistic , pvalue = stats .kruskal(df_classifier[modelos_nao_normal[0]], df_classifier[modelos_nao_normal[1]], df_classifier[modelos_nao_normal[2]])
       print(f'Estatistic, Value_P :  {statistic} {pvalue }')
       alpha =0.95
       if pvalue<= alpha:
             print('H0 rejeitatda ,H1 aceita.  Portanto, estatisticamente os dados são DIFERENTES')
             print('Pode ser aplicado o teste de Equivalnere Tukey')
       else:
             print('H0 aceita ,H1  rejeitada.  Portanto, estatisticamente os dados são IGUAIS')

       import pandas as pd
       
   
       lista_resultados =[]
       
       
       num_metodo = len(modelos_nao_normal)

       for i in range(0,num_metodo):
             lista_resultados.append(df_classifier[modelos_nao_normal[i]])
     
       num_resultados = len(df_classifier[modelos_nao_normal[0]])       

 
       lista_algoritmo = []
       num_metodo = len(modelos_nao_normal) 
       for metodo in range(0,num_metodo):
             for experimento in range(0,num_resultados):
                 lista_algoritmo.append(modelos_nao_normal[metodo])
             
          
       import  numpy as np     
       resultados_algoritmos = {'accuracy': np.concatenate(lista_resultados),
                                 'algoritmo': lista_algoritmo} 

       resultados_df = pd.DataFrame(resultados_algoritmos)
       
       import scikit_posthocs as sp 
       
       import matplotlib.pyplot as plt
       import seaborn as sns
       
       
       fig, axs = plt.subplots(1, 1, figsize=(13,8))
       p_values= sp.posthoc_dunn( resultados_df,  val_col='accuracy', group_col= 'algoritmo')
       
       
      
       
       cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
       heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.87, 0.35, 0.04, 0.3]}
       sp.sign_plot(p_values, **heatmap_args)
       fig.suptitle('Dunn Multiple Comparison Test,p-value', fontsize=16)
       #plt.tight_layout()
       plt.savefig('multiplicomparasion_dnn.jpg', dpi =400, format ='jpg')
       plt.show()
       
       
      
       print( p_values)

       import seaborn as sns 
       sns.heatmap(p_values, annot=True,fmt= '.3f' )
       plt.title('Dunn Multiple Comparison Test,p-value')
       plt.savefig('multiplicomparasion_dnn.heatmap.jpg', dpi =400, format ='jpg')
       plt.show()
       
      
       
       return(0)
   