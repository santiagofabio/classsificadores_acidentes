def classificador_lgbm(x_treino, x_teste, y_treino,y_teste):
     from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
     from sklearn.metrics import  ConfusionMatrixDisplay
     import matplotlib.pyplot as plt
     import lightgbm as lgb
     from datetime import datetime
     dataset = lgb.Dataset(x_treino, label=y_treino)
     parametros={'num_leaves':150,"objective":"binary",'max_depth':7,'learning_rate':0.05,'max_bin':200}

     #inicio = datetime.now()
     lgbm =lgb.train(parametros, dataset)
     #fim =datetime.now()
     #print(f'Tempo de treinamento: {fim-inicio}')
    
     previsoes_lgbm =lgbm.predict(x_teste)
     n_elementos =int(len(previsoes_lgbm))
     for i in range (0,n_elementos):
             if previsoes_lgbm[i]>0.5:
                  previsoes_lgbm[i]= 1
             else:
                 previsoes_lgbm[i]= 0
     
     print(f'Acurrácia lgbm: {accuracy_score(y_teste,previsoes_lgbm)}')
     print(f'Relatorio de Classificação: \n {classification_report(y_teste,previsoes_lgbm)}')
     print(f'Matriz de confusao: \n {confusion_matrix(y_teste,previsoes_lgbm)}')
     ConfusionMatrixDisplay.from_predictions (y_teste, previsoes_lgbm)
     plt.title('Matriz de confusao LGBM')
     plt.savefig('matriz_de_confusao_lgbm.jpg', dpi =300, format ='jpg')
     plt.show()

     return(0)