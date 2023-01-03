def exploracao_quantitativo(df):
    
     import scipy.stats as stats
     import matplotlib.pyplot as plt
     import seaborn as sns 
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
     
     
     figure,( (ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)
     figure.suptitle('BoxPlot')
     sns.boxplot(data=df,x='horas trabalhadas', ax=ax1)
     sns.boxplot(data=df,x='horas dormindas',ax =ax2)
     sns.boxplot(data=df,x='fadiga', ax= ax3)
     sns.boxplot(data=df,x='fss', ax= ax4)
     plt.savefig('boxplot.png',dpi =300, format='jpg') 
    
    
    
    
    
    return(0)