def classificador_regressao_logistica(x_treino, x_teste, y_treino,y_teste):
      from sklearn.linear_model import LogisticRegression
      from sklearn.metrics import  ConfusionMatrixDisplay
      import matplotlib.pyplot as plt
      from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
     
      regressao_logistica =LogisticRegression(random_state=1, max_iter=300, penalty="l2")
      regressao_logistica.fit(x_treino,y_treino )
      print(f'Intercept: {regressao_logistica.intercept_} ')
      print(f'Coef: {regressao_logistica.coef_}')
      previsoes_regressao_logistica = regressao_logistica.predict(x_teste)
     
      print(f'Acurrácia Regressao_logistica: {accuracy_score(y_teste, previsoes_regressao_logistica)}')
      print(f'Relatorio de Classificação: {classification_report(y_teste,previsoes_regressao_logistica)}')
      print(f'Matriz de confusao: {confusion_matrix(y_teste,previsoes_regressao_logistica)}')
      ConfusionMatrixDisplay.from_predictions (y_teste,previsoes_regressao_logistica)
      plt.tight_layout()
      plt.title('Matriz de confusao regressao logistíca')
      plt.savefig('matriz_de_confusao_regressao_logistca.jpg', dpi =300, format ='jpg')
      plt.show() 
     
     
      return(0)