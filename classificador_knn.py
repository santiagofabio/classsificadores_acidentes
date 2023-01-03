def classificador_knn(x_treino, x_teste, y_treino,y_teste):
     from sklearn.neighbors import KNeighborsClassifier 
     from sklearn.metrics import  ConfusionMatrixDisplay
     import matplotlib.pyplot as plt
     from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
     
     
     knn = KNeighborsClassifier(n_neighbors =5, metric = "minkowski")
     knn.fit(x_treino,y_treino )

     previsoes_knn = knn.predict(x_teste)
     
     print('Acurrácia Knn: \n: {:.2f}'.format(accuracy_score(y_teste, previsoes_knn)))
     print('Relatorio de Classificação:')
     print(classification_report(y_teste,previsoes_knn))
     print('Matriz de confusao:')
     print(confusion_matrix(y_teste,previsoes_knn))
     ConfusionMatrixDisplay.from_predictions (y_teste,previsoes_knn)
     plt.tight_layout()
     plt.title('Matriz de confusao KNN ')
     plt.savefig('matriz_de_confusao_knn.jpg', dpi = 300,  format ='jpg')
     plt.show()
     
     
     
     
     return(0)