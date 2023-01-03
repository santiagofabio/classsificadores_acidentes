def visualizacao_recall_precision(x_treino, x_teste, y_treino,y_teste,x_previsores,y_classe):
      from sklearn.naive_bayes import GaussianNB
      from sklearn.metrics import  ConfusionMatrixDisplay
      from sklearn.metrics import roc_curve
      from sklearn.metrics import RocCurveDisplay
      from sklearn.metrics import PrecisionRecallDisplay
      from sklearn.metrics import precision_recall_curve
      from sklearn.metrics import PrecisionRecallDisplay
      import matplotlib.pyplot as plt
      from sklearn.model_selection import train_test_split
      from sklearn.pipeline import make_pipeline
      from sklearn.preprocessing import StandardScaler
      from sklearn.linear_model import LogisticRegression
      from sklearn.metrics import roc_curve
      from sklearn.metrics import RocCurveDisplay
      
      #x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(x_previsores, y_classe, stratify=y_classe)
      
      
      from sklearn.svm import SVC
      
        

      
      clf_nb = GaussianNB()
      clf_nb.fit(x_treino,y_treino)
      y_naive = clf_nb.predict(x_teste)
      fpr_nb, tpr_nb, _ = roc_curve(y_teste,y_naive, pos_label=clf_nb.classes_[1])
      roc_display_nb = RocCurveDisplay(fpr=fpr_nb, tpr=tpr_nb)
      prec_nb, recall_nb, _ = precision_recall_curve(y_teste,y_naive)
      pr_display_nb = PrecisionRecallDisplay(precision=prec_nb, recall=recall_nb)
      
      
      
      
      #------------SVC 
      from sklearn.svm import SVC
      clf_svm =SVC(kernel='rbf', C=1, random_state=1)
      clf_svm.fit(x_treino,y_treino)
      y_svm = clf_svm.predict(x_teste)
      
      fpr_svm, tpr_svm, _ = roc_curve(y_teste,y_svm, pos_label=1)
      roc_display_svm = RocCurveDisplay(fpr=fpr_svm, tpr=tpr_svm)
      prec_svm, recall_svm, _ = precision_recall_curve(y_teste,y_svm) 
      pr_display_svm = PrecisionRecallDisplay(precision=prec_svm, recall=recall_svm)
      #-----------------------------------------
      
      
     # DecisionTreeClassifier
      from sklearn.tree import DecisionTreeClassifier
      clf_arvore = DecisionTreeClassifier(criterion = "entropy", random_state=0 , max_depth=3)
      clf_arvore.fit(x_treino, y_treino)
      y_arvore= clf_arvore.predict(x_teste)
      
      fpr_arvore, tpr_arvore, _ = roc_curve(y_teste,y_arvore, pos_label=1)
      roc_display_arvore = RocCurveDisplay(fpr=fpr_arvore, tpr=tpr_arvore)
      prec_arvore, recall_arvore, _ = precision_recall_curve(y_teste,y_arvore) 
      pr_display_arvore = PrecisionRecallDisplay(precision=prec_arvore, recall=recall_arvore)
      #--------------------------------------------
      
      
      
      #--- KNN
      from sklearn.neighbors import KNeighborsClassifier
      
      clf_knn = KNeighborsClassifier(n_neighbors =5, metric = "minkowski")
      clf_knn.fit(x_treino,y_treino )
      y_knn = clf_knn.predict(x_teste)
      
      fpr_knn, tpr_knn, _ = roc_curve(y_teste,y_knn, pos_label=1)
      roc_display_knn = RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn)
      prec_knn, recall_knn, _ = precision_recall_curve(y_teste,y_knn) 
      pr_display_knn = PrecisionRecallDisplay(precision=prec_knn, recall=recall_knn)
      
      #------LGBM
      import lightgbm as lgb
      dataset = lgb.Dataset(x_treino, label=y_treino)
      parametros={'num_leaves':150,"objective":"binary",'max_depth':7,'learning_rate':0.05,'max_bin':200}

      clf_lgbm =lgb.train(parametros, dataset)
      y_lgbm =clf_lgbm.predict(x_teste)
      n_elementos =int(len(y_lgbm))
      for i in range (0,n_elementos):
             if y_lgbm[i]>0.5:
                  y_lgbm[i]= 1
             else:
                 y_lgbm[i]= 0
    
      fpr_lgbm, tpr_lgbm, _ = roc_curve(y_teste,y_lgbm, pos_label=1)
      roc_display_lgbm = RocCurveDisplay(fpr=fpr_lgbm, tpr=tpr_lgbm)
      prec_lgbm, recall_lgbm, _ = precision_recall_curve(y_teste,y_knn) 
      pr_display_lgbm = PrecisionRecallDisplay(precision=prec_lgbm, recall=recall_lgbm)
       
      
      #-----RANDOM FOREST
      from sklearn.ensemble import RandomForestClassifier
      clf_random_forest =  RandomForestClassifier(n_estimators= 150,criterion = "gini", random_state=0, max_depth=4)
      clf_random_forest.fit(x_treino,y_treino)
      y_random_forest =clf_random_forest.predict(x_teste)
      
      fpr_random_forest, tpr_random_forest, _ = roc_curve(y_teste,y_random_forest, pos_label=1)
      roc_display_random_forest = RocCurveDisplay(fpr=fpr_random_forest, tpr=tpr_random_forest)
      prec_random_forest, recall_random_forest, _ = precision_recall_curve(y_teste,y_random_forest) 
      pr_display_random_forest = PrecisionRecallDisplay(precision=prec_random_forest, recall=recall_random_forest)

      
      #Regressao Logistica
      from sklearn.linear_model import LogisticRegression
      clf_regressao_logistica =LogisticRegression(random_state=1, max_iter=300, penalty="l2")
      clf_regressao_logistica.fit(x_treino,y_treino )
      y_rl = clf_regressao_logistica.predict(x_teste)
      fpr_rl, tpr_lr, _ = roc_curve(y_teste,y_rl, pos_label=1)
      roc_display_lr = RocCurveDisplay(fpr=fpr_rl, tpr=tpr_lr)
      prec_rl, recall_rl, _ = precision_recall_curve(y_teste,y_rl) 
      pr_display_rl = PrecisionRecallDisplay(precision=prec_rl, recall=recall_rl)
    
      
      # XGBOOST
      
      from xgboost import XGBClassifier
      clf_xgboost =XGBClassifier(max_depth =4, n_estimators =200, random_state =3 , learning_rate =0.05)
      clf_xgboost.fit(x_treino,y_treino)
      y_xgboost =clf_xgboost.predict(x_teste)
      
      fpr_xgboost, tpr_xgboost, _ = roc_curve(y_teste,y_xgboost, pos_label=1)
      roc_display_xgboost = RocCurveDisplay(fpr=fpr_xgboost, tpr=tpr_xgboost)
      prec_xgboost, recall_xgboost, _ = precision_recall_curve(y_teste,y_xgboost) 
      pr_display_xgboost = PrecisionRecallDisplay(precision=prec_xgboost, recall=recall_xgboost)
       
      
      
           
      import matplotlib.pyplot as plt

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
      roc_display_nb.plot(ax=ax1, label ='Naive Bayes', marker = 'o')
      roc_display_svm.plot(ax=ax1, label ='SVM', marker ='x')
      roc_display_arvore.plot(ax=ax1, label ='Decision Tree' , marker = '^')
      roc_display_knn.plot(ax=ax1, label ='KNN', marker = 'v')
      roc_display_lgbm.plot(ax=ax1, label ='LGBM', marker ='H')
      roc_display_random_forest.plot(ax=ax1, label ='Random Forest', marker = '<')
      roc_display_lr.plot(ax=ax1, label ='Logistic Regression', marker ='D')
      roc_display_xgboost.plot(ax=ax1, label ='Xgboost', marker ='p')
      plt.legend(loc ='best')
      
  
     
      
      
      
      pr_display_nb.plot(ax=ax2,label ='Naive Bayes', marker = 'o')
      pr_display_svm.plot(ax=ax2, label ='SVM', marker ='x')
      pr_display_arvore.plot(ax=ax2, label ='Decision Tree', marker = '^')
      pr_display_knn.plot(ax=ax2, label ='KNN', marker = 'v')
      pr_display_lgbm.plot(ax=ax2, label ='LGBM', marker ='H')
      pr_display_random_forest.plot(ax=ax2, label ='Random Forest', marker = '<')
      pr_display_rl.plot(ax=ax2, label ='Logistic Regression', marker ='D')
      pr_display_xgboost.plot(ax=ax2, label ='Xgboost', marker ='p')
      plt.tight_layout()
      
      
     
      plt.legend(loc ='best')
      plt.savefig('Visuaizao_precision_recall.jpg', dpi =300, format ='jpg')
      plt.show()
      

      
      """
       
      
      
      
      
      
      #--- KNN
     
      
      clf_knn = 
      clf_knn.fit(x_treino,y_treino )
      y_knn = clf_knn.predict(x_teste)
      
      fpr_knn, tpr_knn, _ = roc_curve(y_teste,y_knn, pos_label=1)
      roc_display_knn = RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn)
      prec_knn, recall_knn, _ = precision_recall_curve(y_teste,y_knn) 
      pr_display_knn = PrecisionRecallDisplay(precision=prec_knn, recall=recall_knn)
      
      #------LGBM
      import lightgbm as lgb
      dataset = lgb.Dataset(x_treino, label=y_treino)
      parametros={'num_leaves':150,"objective":"binary",'max_depth':7,'learning_rate':0.05,'max_bin':200}

      clf_lgbm =lgb.train(parametros, dataset)
      y_lgbm =clf_lgbm.predict(x_teste)
      n_elementos =int(len(y_lgbm))
      for i in range (0,n_elementos):
             if y_lgbm[i]>0.5:
                  y_lgbm[i]= 1
             else:
                 y_lgbm[i]= 0
    
      fpr_lgbm, tpr_lgbm, _ = roc_curve(y_teste,y_lgbm, pos_label=1)
      roc_display_lgbm = RocCurveDisplay(fpr=fpr_lgbm, tpr=tpr_lgbm)
      prec_lgbm, recall_lgbm, _ = precision_recall_curve(y_teste,y_knn) 
      pr_display_lgbm = PrecisionRecallDisplay(precision=prec_lgbm, recall=recall_lgbm)
       
      
      #-----RANDOM FOREST
      from sklearn.ensemble import RandomForestClassifier
      clf_random_forest =  RandomForestClassifier(n_estimators= 150,criterion = "gini", random_state=0, max_depth=4)
      clf_random_forest.fit(x_treino,y_treino)
      y_random_forest =clf_random_forest.predict(x_teste)
      
      fpr_random_forest, tpr_random_forest, _ = roc_curve(y_teste,y_random_forest, pos_label=1)
      roc_display_random_forest = RocCurveDisplay(fpr=fpr_random_forest, tpr=tpr_random_forest)
      prec_random_forest, recall_random_forest, _ = precision_recall_curve(y_teste,y_random_forest) 
      pr_display_random_forest = PrecisionRecallDisplay(precision=prec_random_forest, recall=recall_random_forest)

      
      #Regressao Logistica
      from sklearn.linear_model import LogisticRegression
      clf_regressao_logistica =LogisticRegression(random_state=1, max_iter=300, penalty="l2")
      clf_regressao_logistica.fit(x_treino,y_treino )
      y_rl = clf_regressao_logistica.predict(x_teste)
      fpr_rl, tpr_lr, _ = roc_curve(y_teste,y_rl, pos_label=1)
      roc_display_lr = RocCurveDisplay(fpr=fpr_rl, tpr=tpr_lr)
      prec_rl, recall_rl, _ = precision_recall_curve(y_teste,y_rl) 
      pr_display_rl = PrecisionRecallDisplay(precision=prec_rl, recall=recall_rl)
    
      
      # XGBOOST
      
      from xgboost import XGBClassifier
      clf_xgboost =XGBClassifier(max_depth =4, n_estimators =200, random_state =3 , learning_rate =0.05)
      clf_xgboost.fit(x_treino,y_treino)
      y_xgboost =clf_xgboost.predict(x_teste)
      
      fpr_xgboost, tpr_xgboost, _ = roc_curve(y_teste,y_xgboost, pos_label=1)
      roc_display_xgboost = RocCurveDisplay(fpr=fpr_xgboost, tpr=tpr_xgboost)
      prec_xgboost, recall_xgboost, _ = precision_recall_curve(y_teste,y_xgboost) 
      pr_display_xgboost = PrecisionRecallDisplay(precision=prec_xgboost, recall=recall_xgboost)
       
      
      
           
      import matplotlib.pyplot as plt

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
      roc_display_nb.plot(ax=ax1, label ='Naive Bayes', marker = 'o')
      roc_display_svm.plot(ax=ax1, label ='SVM', marker ='x')
      roc_display_arvore.plot(ax=ax1, label ='Decision Tree' , marker = '^')
      roc_display_knn.plot(ax=ax1, label ='KNN', marker = 'v')
      roc_display_lgbm.plot(ax=ax1, label ='LGBM', marker ='H')
      roc_display_random_forest.plot(ax=ax1, label ='Random Forest', marker = '<')
      roc_display_lr.plot(ax=ax1, label ='Logistic Regression', marker ='D')
      roc_display_xgboost.plot(ax=ax1, label ='Xgboost', marker ='p')
      plt.tight_layout()
      plt.legend(loc ='best')
      
  
     
      
      
      
      pr_display_nb.plot(ax=ax2,label ='Naive Bayes', marker = 'o')
      pr_display_svm.plot(ax=ax2, label ='SVM', marker ='x')
      pr_display_arvore.plot(ax=ax2, label ='Decision Tree', marker = '^')
      pr_display_knn.plot(ax=ax2, label ='KNN', marker = 'v')
      pr_display_lgbm.plot(ax=ax2, label ='LGBM', marker ='H')
      pr_display_random_forest.plot(ax=ax2, label ='Random Forest', marker = '<')
      pr_display_rl.plot(ax=ax2, label ='Logistic Regression', marker ='D')
      pr_display_xgboost.plot(ax=ax2, label ='Xgboost', marker ='p')
      plt.tight_layout()
      
      
     
      plt.legend(loc ='best')
      plt.savefig('Visuaizao_precision_recall.jpg', dpi =300, format ='jpg')
      plt.show()
      
      """
          
      
     
     
      return(0)