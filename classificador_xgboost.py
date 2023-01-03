def classificador_xgboost(x_treino, x_teste, y_treino,y_teste):
     from xgboost import XGBClassifier
     
     from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay
     import matplotlib.pyplot as plt 
     xgboost =XGBClassifier(max_depth =4, n_estimators =200, random_state =3 , learning_rate =0.05)
     xgboost.fit(x_treino,y_treino)
     previsoes_xgboost =xgboost.predict(x_teste)
     print('Acuracia XGBoost: {:.2f}'.format(accuracy_score(y_teste,previsoes_xgboost)))
     print(f'Matriz de confusao')
     print(confusion_matrix(y_teste,previsoes_xgboost))
     print(classification_report(y_teste,previsoes_xgboost))
     ConfusionMatrixDisplay.from_predictions(y_teste, previsoes_xgboost)
     plt.show()
     return(0)