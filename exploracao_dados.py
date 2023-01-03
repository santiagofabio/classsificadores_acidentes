def exploracao_dados(df):
     # Catalogo dos dados
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
      
      #---------QULALITATIVOS-----------------
      import matplotlib.pyplot as plt
      import seaborn as sns 
    
      # habitos de vida. 
      #Analisar contagem 
      print(df['fumante'].value_counts())
      sns.countplot(data =df , x ='fumante').set_title('Fumante')
      plt.show()
      
      print(df['bebidas'].value_counts())
      sns.countplot(data =df , x ='bebidas',hue='bebidas').set_title('Uso de bebidas')
      plt.show()
      
      print(df['drogas'].value_counts())
      sns.countplot(data =df , x ='drogas', hue='drogas').set_title('Uso de drogas')
      plt.show()
      
      # condições de trabalho
      print(df['trabalho excessivo'].value_counts())
      sns.countplot(data =df , x ='trabalho excessivo',hue = 'trabalho excessivo').set_title('Trabalho excessivo')
      plt.show()
      
      print(df['posicao incomoda'].value_counts())
      sns.countplot(data =df , x ='posicao incomoda', hue ='trabalho excessivo').set_title('posicao incomoda')
      plt.show()
      
      
      from matplotlib.pylab import rcParams
      rcParams['figure.figsize']= 15,10
      fig, ((ax1, ax2,ax3)) = plt.subplots(1, 3)
      fig.suptitle('Healthy habits', fontsize =16)
      sns.countplot(data =df , x ='fumante', ax =ax1,hue='fumante' ).set_xticklabels([],rotation=30, fontsize =14) 
      sns.countplot(data =df , x ='bebidas', ax =ax2,  hue='bebidas').set_xticklabels([ ],rotation=30,fontsize =14) 
      sns.countplot(data =df , x ='drogas', ax =ax3,  hue='drogas').set_xticklabels([ ],rotation=30,fontsize =14) 
      plt.yticks(fontsize =14)
      plt.xticks(fontsize =14)
      fig.savefig('habitos_de_vida.jpg', dpi=300, format='jpg')
      
      #sns.countplot(data =df , x ='posicao incomoda', ax =ax3, hue='posicao incomoda' )
      plt.show()
      
      fig, ((ax1, ax2)) = plt.subplots(1, 2)
      fig.suptitle('Condicoes de trabalho', fontsize =16)
      sns.countplot(data =df , x ='trabalho excessivo', ax =ax1,hue='trabalho excessivo' ).set_xticklabels([],rotation=30,fontsize =14) 
      sns.countplot(data =df , x ='posicao incomoda', ax =ax2,  hue='posicao incomoda').set_xticklabels([ ],rotation=30,fontsize =14) 
      plt.yticks(fontsize =14)
      plt.xticks(fontsize =14)
      fig.savefig('condicoes_de_trabalho.jpg', dpi=300, format='jpg')
      plt.show()
   
      return(0)

      
      
    
     
     
     
     
  