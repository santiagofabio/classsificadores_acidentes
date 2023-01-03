def classficador_svm(x_treino, x_teste, y_treino,y_teste):
     from sklearn.svm import SVC
     from sklearn.metrics import  ConfusionMatrixDisplay
     import matplotlib.pyplot as plt
     from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
     svm =SVC(kernel='rbf', C=1, random_state=1)
     svm.fit(x_treino,y_treino)
     previsoes_svm = svm.predict(x_teste)
     
     print(f'Acurrácia SVM: {accuracy_score(y_teste, previsoes_svm)}')
     print(f'Relatorio de Classificação: \n {classification_report(y_teste, previsoes_svm)}')
     print(f'Matriz de confusao: \n {confusion_matrix(y_teste, previsoes_svm)}')
     ConfusionMatrixDisplay.from_predictions (y_teste, previsoes_svm,values_format = '.2g' )
     plt.title('Matriz de confusao SVM')
     plt.savefig('matriz_de_confusao_svm.jpg', dpi =300, format ='jpg')
     plt.show() 
     return(0)