def preprocessamento_dados2(df):
     from  sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
     from sklearn.compose  import ColumnTransformer
       
     # - Aplicando LabelEnconder 
     """
     Catalogo de dados 
     fumante                object [0] ok
     bebidas                object [1] ok 
     drogas                 object [2] ok
     trabalho excessivo     object [3] ok
     posicao incomoda       object [4] ok
     horas dormindas       float64 [5] 
     fss                   float64 [6]
     fadiga                float64 [7]
     horas trabalhadas     float64 [8]
     ess                   float64 [9] 
     acidentes             float64 [10]
     """
     
     
     df['bebidas'].replace({'>4 vezes por semana':'>4 week'}, inplace=True  )
     df['bebidas'].replace({'2-3 vezes por semana':'2-3 week'}, inplace=True  )
     df['bebidas'].replace({'Diariamente':'daily'}, inplace=True  )
     df['bebidas'].replace({'Não':'Not'}, inplace=True  )
     
     
     
     #-Previsores
     previsores = df.iloc[:,0:10].values
     #print(previsores[0:5])
     previsores[:,0] = LabelEncoder().fit_transform(previsores[:,0])
     previsores[:,1] = LabelEncoder().fit_transform(previsores[:,1])
     previsores[:,2] = LabelEncoder().fit_transform(previsores[:,2])
     previsores[:,3] = LabelEncoder().fit_transform(previsores[:,3])
     previsores[:,4] = LabelEncoder().fit_transform(previsores[:,4])
     
     #Clase
     classe_alvo =df.iloc[:, 10].values
     # print( classe_alvo[0:5])
     
     #Aplica OneHotEncoder()
     previsores2 = ColumnTransformer(transformers =[('Onehot', OneHotEncoder(),[0,1,2,3,4] )], 
                                remainder ='passthrough').fit_transform(previsores)
     
     # Aplica Scalonamento

     previsores_escalonado =StandardScaler().fit_transform(previsores2)
     
     import pickle 
     # Salava os arquivos 
     with open('previsores_escalonados.pkl','wb') as arquivo:
            pickle.dump(previsores_escalonado, arquivo)

     with open('classe_alvo.pkl','wb') as arquivo:
             pickle.dump(classe_alvo, arquivo)
    
    
    
     return(0)

