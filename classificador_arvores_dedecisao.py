def classificador_arvores_dedecisao(x_treino, x_teste, y_treino,y_teste):
     from sklearn.tree import DecisionTreeClassifier
     from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
     import matplotlib.pyplot as plt
     arvore = DecisionTreeClassifier(criterion = "entropy", random_state=0 , max_depth=3)
     arvore.fit(x_treino, y_treino)
     previsoes = arvore.predict(x_teste)
        
     print("Acurracia: {:.2f}".format(accuracy_score(y_teste,previsoes))) 
     print("Matriz de confusão:")
     print(confusion_matrix(y_teste,previsoes ))
     ConfusionMatrixDisplay.from_predictions(y_teste,previsoes)
     plt.title('Matriz de confusao arvore de decisão')
     plt.savefig('matriz_de_confusao_arvore.jpg', dpi =300, format ='jpg')
     plt.show()
     
     
     return(0)