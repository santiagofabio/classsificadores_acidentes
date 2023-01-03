def exploracao_quantitativo(df):
    
     import scipy.stats as stats
     import matplotlib.pyplot as plt
     import seaborn as sns 
     
     
     #label_quantitativo =['Sleeping hours' ,'FSS','Fatigue','Worked hours' , 'ESS'  ]
     
     label_quantitativo_horas =['Sleeping hours','Worked hours' ]
     
     label_quantitativo_pontos =['ESS' ,'FSS','Fatigue']
     
     
     df2_quatitativo= df.iloc[:,6:11]
     from matplotlib.pylab import rcParams
     rcParams['figure.figsize']= 15,10
     """
     fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)
     fig.suptitle('Dados quanitativos', fontsize =16)
     sns.histplot(data =df , x =label_quantitativo[0], ax =ax1, kde=True,  stat='density')
     sns.histplot(data =df , x =label_quantitativo[1], ax =ax2, kde=True,  stat='density')
     sns.histplot(data =df , x =label_quantitativo[2], ax =ax3, kde=True,  stat='density' ) 
     sns.histplot(data =df , x =label_quantitativo[3], ax =ax4, kde=True,  stat='density' )
     fig.savefig('metricas.jpg', dpi=300, format='jpg')
    
     plt.show() 
     

     
     """
     
     
         #----------------- BOXPLOT----------------
     plt.tight_layout()
     sns.boxplot(data=df[label_quantitativo_horas], orient="h",bootstrap=100  ).set_yticklabels(label_quantitativo_horas,rotation=30, fontsize=14) 
     plt.title('Boxplot Quantitative data by hours', fontsize=16)
     plt.xlabel('Hours', fontsize=14)
     plt.savefig('boxplot_dados_quatitativos_horas.jpg',dpi =400, format='jpg') 
     plt.show()
     plt.close()
      
      #----------------- BOXPLOT----------------
     plt.tight_layout()
     sns.boxplot(data=df[label_quantitativo_pontos], orient="h",bootstrap=100  ).set_yticklabels(label_quantitativo_pontos,rotation=30, fontsize=14) 
     plt.title('Boxplot Quantitative data by points', fontsize=16)
     plt.xlabel('Hours', fontsize=14)
     plt.savefig('boxplot_dados_quatitativos_points.jpg',dpi =400, format='jpg') 
     plt.show()
     plt.close()
      
     
     
     
     
     """ 
      fumante                object
      bebidas                object
      drogas                 object
      acidentes               int64
      trabalho excessivo     object
      posicao incomoda       object
      horas dormindas       float64
      fss                     int64
      fadiga                  int64
      horas trabalhadas     float64
      ess                     int64
      """
     
   
     """
    
     #mapeamento de correlaçoes
     import seaborn as sns 
     import matplotlib.pyplot as plt 
     sns.pairplot(df2_quatitativo)
     plt.savefig('mapeamento_correlacoes.jpg',dpi=300, format='jpg')
     plt.show()
     
     correlacoes=   df2_quatitativo.corr()
     plt.figure()
     sns.heatmap(correlacoes, annot =True)
     plt.savefig('heatmap.jpg', dpi =300,format ='jpg')
     plt.show()
     
     
     figure,( (ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)
     figure.suptitle('BoxPlot')
     sns.boxplot(data=df,x='horas trabalhadas', ax=ax1)
     sns.boxplot(data=df,x='ess',ax =ax2)
     sns.boxplot(data=df,x='fadiga', ax= ax3)
     sns.boxplot(data=df,x='fss', ax= ax4)
     plt.savefig('boxplot.png',dpi =300, format='jpg') 
     plt.show()
     
     
     
     #Analise grafica de normalidade.
     plt.rcParams['figure.figsize'] = (10, 10)
     import scipy.stats as stats
     figure,( (ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)
     figure.suptitle('Nomral Q-Q Plot')
     stats.probplot(df2_quatitativo['fss'], dist='norm', plot=ax1 )
     ax1. set_title('FSS') 
     
     stats.probplot(df2_quatitativo['fadiga'], dist='norm', plot=ax2) 
     ax2.set_title('FADIGA') 
    
     stats.probplot(df2_quatitativo['horas trabalhadas'], dist='norm', plot= ax3)
     ax3.set_title('HORAS TRAB.')
    
     stats.probplot(df2_quatitativo['ess'], dist='norm', plot= ax4)
     ax4.set_title('ESS')
     
     plt.savefig('NormalQQ.png',dpi =300, format='jpg')

     """
     
   

     
     
     
     
     

     
    
    
    
    
     return(0)