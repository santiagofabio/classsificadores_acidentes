def classificador_random_forest(x_treino, x_teste, y_treino,y_teste):
     from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
     import matplotlib.pyplot as plt 
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report, confusion_matrix
     random_forest =  RandomForestClassifier(n_estimators= 150,criterion = "gini", random_state=0, max_depth=4)
     random_forest.fit(x_treino,y_treino)
    
     previsao =random_forest.predict(x_treino)
     print('Acuracia random forest dados de treino:{:.2f}'.format(accuracy_score(y_treino,previsao)))
     previsao =random_forest.predict(x_teste)
     print('Acuracia  random forest dados de teste:{:.2f}'.format(accuracy_score(y_teste,previsao)))
     print(confusion_matrix(y_teste,previsao))
     print(classification_report(y_teste,previsao))
     ConfusionMatrixDisplay.from_predictions(y_teste,previsao)
     plt.title('Matriz de confusao random forest')
     plt.savefig('matriz_de_confusao_random_forest.jpg', dpi =300, format ='jpg')
     plt.show()
     return(0)