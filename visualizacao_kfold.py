def visualizacao_kfold(df_classifier, modelos_disribuicao_normal,modelos_nao_normal):
      
      from pylab import rcParams
      import seaborn as sns 
      import matplotlib.pyplot as plt
      from pylab import rcParams
      from pylab import rcParams
      rcParams['figure.figsize'] = 15, 10
      
      classifier_name=['Naive Bayes', 'KNN', 'Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression','Lightgbm', 'XGBClassifier']
      
      
      print(df_classifier.describe())
   
      
      
      
      for i in range(0,5):
           print( i, modelos_disribuicao_normal[i])
      import statsmodels.api as sm
      
      
      rcParams['figure.figsize'] = 15, 10 
      import scipy.stats as stats
      figure,( (ax1, ax2,ax3) ) = plt.subplots (1,3)
      stats.probplot(x= df_classifier[modelos_nao_normal[0]], dist='norm', plot=ax1 )
      ax1. set_title(modelos_nao_normal[0]) 
      plt.tight_layout()
     
      stats.probplot(df_classifier[modelos_nao_normal[1]], dist='norm', plot=ax2 )
      ax2. set_title(modelos_nao_normal[1])
      plt.tight_layout()     
          
      
      stats.probplot(df_classifier[modelos_nao_normal[2]], dist='norm', plot=ax3 )
      ax3.set_title(modelos_nao_normal[2])
      plt.tight_layout()
      
      #figure.suptitle('Normal QQ Plot ')
      plt.savefig('NormalQQ_Classifier_FALHA.jpg',dpi =300, format='jpg')
      plt.show()
      plt.close()
       
      #-------------- Diagramas Normal QQ -Plot 
        
      #Analise grafica de normalidade.
      plt.rcParams['figure.figsize'] = (15, 10)
      import scipy.stats as stats
      
      figure,( (ax1, ax2,ax3) ) = plt.subplots (1,3)
      stats.probplot(x= df_classifier[modelos_disribuicao_normal[0]], dist='norm', plot=ax1 )
      ax1. set_title(modelos_disribuicao_normal[0]) 
      plt.tight_layout()
     
      stats.probplot(df_classifier[modelos_disribuicao_normal[1]], dist='norm', plot=ax2 )
      ax2. set_title(modelos_disribuicao_normal[1])
      plt.tight_layout()     
          
      
      stats.probplot(df_classifier[modelos_disribuicao_normal[2]], dist='norm', plot=ax3 )
      ax3.set_title(modelos_disribuicao_normal[2])
      plt.tight_layout()
      
     
      plt.savefig('NormalQQ_Classifier_linha.jpg',dpi =300, format='jpg')
      plt.show()
      plt.close()
      
       #Analise grafica de normalidade.
      plt.rcParams['figure.figsize'] = (15, 10)
      import scipy.stats as stats
      
      figure,( (ax1, ax2) ) = plt.subplots (1,2)
      stats.probplot(x= df_classifier[modelos_disribuicao_normal[3]], dist='norm', plot=ax1 )
      ax1. set_title(modelos_disribuicao_normal[3]) 
      plt.tight_layout()
     
      stats.probplot(df_classifier[modelos_disribuicao_normal[4]], dist='norm', plot=ax2 )
      ax2. set_title(modelos_disribuicao_normal[4])
      plt.tight_layout()     
      plt.savefig('NormalQQ_Classifier_linha_p2.jpg',dpi =300, format='jpg')
      plt.show()
      plt.close()     
      
   
      
      
      
      
      #----------------- BOXPLOT----------------
      plt.tight_layout()
      sns.boxplot(data=df_classifier[classifier_name], orient="h",bootstrap=100  ).set_yticklabels(['GNB','KNN','Decison Tree','Random Forest','SVM', 'RL','LGBM' , 'XGB' ],rotation=30, fontsize=14) 
      plt.title('Boxplot distribuicao Accuracy Kfold', fontsize=16)
      plt.xlabel('Acurracy' , fontsize=14)                                                                                                   
      #plt.xlim(0.77, 0.85)
      plt.savefig('boxplot_classidicador.jpg',dpi =400, format='jpg') 
      plt.show()
      plt.close()
      
      #----------------- KDEPLOT----------------
      plt.tight_layout()
      sns.kdeplot(data =df_classifier, x =classifier_name[0], label =classifier_name[0],  marker = 'o')
      sns.kdeplot(data =df_classifier, x = classifier_name[1], label =classifier_name[1], marker = 'v')
      sns.kdeplot(data =df_classifier, x = classifier_name[2], label =classifier_name[2],  marker = '^')
      sns.kdeplot(data =df_classifier, x =classifier_name[3] , label =classifier_name[3],marker = '<'  ) 
      sns.kdeplot(data =df_classifier, x =classifier_name[4],  label =classifier_name[4], marker ='x')
      sns.kdeplot(data =df_classifier, x =classifier_name[5], label =classifier_name[5], marker ='D')
      sns.kdeplot(data =df_classifier, x =classifier_name[6], label =classifier_name[6],marker ='H')
      sns.kdeplot(data =df_classifier, x =classifier_name[7] , label =classifier_name[7], marker ='p' )
      
      plt.xlabel('Acurracy' , fontsize=14) 
      plt.legend(loc ='best' , fontsize =14)
      
      plt.title('Classifier distribution', fontsize=16) 
      plt.savefig('distribution_classifier.jpg', dpi =400, format = 'jpg')
      plt.show() 
      plt.close() 
       
      
      

      
      
      return(0)
      
      
      
      
      
      
      
      
      
      
      
      
      
      

      

      
      
      
      
      
     