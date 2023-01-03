def vizualizacao_confusion_matrix(x_treino, x_teste, y_treino,y_teste): 
      from sklearn.svm import SVC
      from sklearn.metrics import  ConfusionMatrixDisplay, confusion_matrix, accuracy_score,classification_report 
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.neighbors import KNeighborsClassifier 
      from sklearn.naive_bayes import GaussianNB
      import lightgbm as lgb
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.linear_model import LogisticRegression
      dataset = lgb.Dataset(x_treino, label=y_treino)
      from sklearn.svm import SVC
      from xgboost import XGBClassifier
      import lightgbm as lgbm
      import matplotlib.pyplot as plt
      dataset = lgb.Dataset(x_treino, label=y_treino)
      parametros={'num_leaves':150,"objective":"binary",'max_depth':7,'learning_rate':0.05,'max_bin':200}
      from pylab import rcParams
      rcParams['figure.figsize'] = 15, 10
                          
      figure, ((ax1, ax2, ax3, ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(2, 4, figsize=(15, 10))
      plt.tight_layout()
     
      modelo_nb = GaussianNB()
      modelo_nb.fit(x_treino,y_treino)
      y_pred=  modelo_nb.predict(x_teste)
      cf_matrix = confusion_matrix(y_teste, y_pred )
      print('Acurrácia GaussianNB:{:.2f}'.format(accuracy_score(y_teste,  y_pred)))
      print(classification_report(y_teste,y_pred) )
      disp = ConfusionMatrixDisplay(cf_matrix)
      disp.plot(ax =ax1, xticks_rotation= 45)
      disp.ax_.set_title("Naive Bayes",fontsize=14)
      disp.im_.colorbar.remove()
      
     
      modelo_knn = KNeighborsClassifier(n_neighbors =5, metric = "minkowski")
                 
      modelo_knn.fit(x_treino,y_treino)
      y_pred_knn = modelo_knn.predict(x_teste)
      cf_matrix = confusion_matrix(y_teste, y_pred_knn)
      print('Acurrácia KNN :{:.2f}'.format(accuracy_score(y_teste,y_pred_knn)))
      print(classification_report(y_teste,y_pred_knn) )
      disp = ConfusionMatrixDisplay(cf_matrix)
      disp.plot(ax =ax2, xticks_rotation= 45)
      disp.ax_.set_title("KNN",fontsize=14)
      disp.im_.colorbar.remove()
     
     
      modelo_arvore =DecisionTreeClassifier(criterion = "entropy", random_state=0 , max_depth=3)
      modelo_arvore.fit(x_treino,y_treino)
      y_pred_arvore = modelo_arvore.predict(x_teste)
      print('Acurrácia DecisionTree :{:.2f}'.format(accuracy_score(y_teste,y_pred_arvore)))
      print(classification_report(y_teste, y_pred_arvore) )
      cf_matrix = confusion_matrix(y_teste, y_pred_arvore)
      disp = ConfusionMatrixDisplay(cf_matrix)
      disp.plot(ax =ax3, xticks_rotation= 45)
      disp.ax_.set_title("Decision Tree",fontsize=14)
      disp.im_.colorbar.remove()
     
     
      from sklearn.ensemble import RandomForestClassifier
      random_forest =  RandomForestClassifier(n_estimators= 150,criterion = "gini", random_state=0, max_depth=4)
      random_forest.fit(x_treino,y_treino)
      y_random =random_forest.predict(x_teste)
      print('Acurrácia RandomForest :{:.2f}'.format(accuracy_score(y_teste, y_random)))
      print(classification_report(y_teste,  y_random) )
      cf_matrix = confusion_matrix(y_teste,y_random)
      disp = ConfusionMatrixDisplay(cf_matrix)
      disp.plot(ax =ax4, xticks_rotation= 45)
      disp.ax_.set_title("Random Forest",fontsize=14)
      disp.im_.colorbar.remove()
     
     
     
      modelo_svc = SVC(kernel='rbf', C=1, random_state=1)
      modelo_svc.fit(x_treino,y_treino)
      y_pred_svc = modelo_svc.predict(x_teste)
      print('Acurrácia SVM :{:.2f}'.format(accuracy_score(y_teste,y_pred_svc)))
      print(classification_report(y_teste,y_pred_svc) )
      cf_matrix = confusion_matrix(y_teste, y_pred_svc)
      disp = ConfusionMatrixDisplay(cf_matrix)
      disp.plot(ax =ax5, xticks_rotation= 45)
      disp.ax_.set_title("SVM",fontsize=14)
      disp.im_.colorbar.remove()
     
      regressao_logistica =LogisticRegression(random_state=1, max_iter=300, penalty="l2")
      regressao_logistica.fit(x_treino,y_treino )
      y_regressao_logistica=  regressao_logistica.predict(x_teste)
      print('Acurrácia Logstic Regressor :{:.2f}'.format(accuracy_score(y_teste,y_regressao_logistica)))
      print(classification_report(y_teste, y_regressao_logistica) )
     
      cf_matrix = confusion_matrix(y_teste,y_regressao_logistica)
      disp = ConfusionMatrixDisplay(cf_matrix)
      disp.plot(ax =ax6, xticks_rotation= 45)
      disp.ax_.set_title("Logistic Regression",fontsize=14)
      disp.im_.colorbar.remove()

     
      import lightgbm as lgb
      dataset = lgb.Dataset(x_treino, label=y_treino)
      parametros={'num_leaves':150,"objective":"binary",'max_depth':7,'learning_rate':0.05,'max_bin':150 , 'verbose':-1} 
      lgbm =lgb.train(parametros, dataset)
      previsoes_lgbm =lgbm.predict(x_teste)
      n_elementos =int(len(previsoes_lgbm))
      for i in range (0,n_elementos):
              if previsoes_lgbm[i]>0.5:
                  previsoes_lgbm[i]= 1
              else:
                 previsoes_lgbm[i]= 0
                
     
     
      print('Acurrácia lightgbm:{:.2f}'.format(accuracy_score(y_teste,previsoes_lgbm)))
      print(classification_report(y_teste, previsoes_lgbm) )
      cf_matrix = confusion_matrix(y_teste,previsoes_lgbm)
      disp = ConfusionMatrixDisplay(cf_matrix)
      disp.plot(ax =ax7, xticks_rotation= 45)
      disp.ax_.set_title("Lightgbm",fontsize=14)
      disp.im_.colorbar.remove()
      
     
     

     
    
      from xgboost import XGBClassifier 
      xgboost =XGBClassifier(max_depth =4, n_estimators =200)
    
      xgboost.fit(x_treino,y_treino)
      y_xgboost= xgboost.predict(x_teste)
      print('Acurrácia Xgboost :{:.2f}'.format(accuracy_score(y_teste,y_xgboost)))
      print(classification_report(y_teste, y_xgboost) )
     
      cf_matrix = confusion_matrix(y_teste,y_xgboost)
      disp = ConfusionMatrixDisplay(cf_matrix)
      disp.plot(ax =ax8, xticks_rotation= 45)
      disp.ax_.set_title("XGBClassifier",fontsize=14)
      disp.im_.colorbar.remove()
     
 
     
     

     
     
      figure.colorbar(disp.im_,ax=[ ax1, ax2, ax3, ax4, ax5,ax6, ax7, ax8]) 
     
      figure.suptitle('Confusion Matrix Classifier', fontsize=16)
      plt.savefig('confusion_matrix_classifiers.jpg', dpi =300,format ='jpg')
      plt.show()
     
      return(0)         
     