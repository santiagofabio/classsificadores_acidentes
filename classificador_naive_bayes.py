def classificador_naive_bayes(x_treino, x_teste, y_treino,y_teste):
      from sklearn.naive_bayes import GaussianNB
      from sklearn.metrics import  ConfusionMatrixDisplay
      import matplotlib.pyplot as plt
      naive =GaussianNB()
      naive.fit(x_treino,y_treino) 
      previsoes_naive = naive.predict(x_teste)
      from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
     
      print('Acurrácia:{:.2f}'.format(accuracy_score(y_teste,previsoes_naive)))
      print(f'Relatorio de Classificação: \n {classification_report(y_teste,previsoes_naive)}')
      print(f'Matriz de confusao:')
      print( confusion_matrix(y_teste,previsoes_naive))
      ConfusionMatrixDisplay.from_predictions (y_teste,previsoes_naive)
      plt.title('Matriz de confusão Naive Bayes')
      plt.savefig('matriz_confusao_naive_bayes.jpg', dpi =300, format ='jpg' )
      plt.show()
     
      from sklearn.metrics import roc_curve
      from sklearn.metrics import RocCurveDisplay
      from sklearn.metrics import PrecisionRecallDisplay
      from sklearn.metrics import precision_recall_curve
      from sklearn.metrics import PrecisionRecallDisplay
     
      y_score = naive.predict(x_teste) 
      fpr, tpr, _ = roc_curve(y_teste, y_score, pos_label=1)
      roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
      prec, recall, _ = precision_recall_curve(y_teste, y_score)
      pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
           
      import matplotlib.pyplot as plt

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
      roc_display.plot(ax=ax1, label ='Naive Bayes')
      pr_display.plot(ax=ax2)
      plt.legend(loc ='best')
      plt.show()
     
     
      return(0)