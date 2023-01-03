def reducao_dimensionalidade(previsores_escalodados):
    from sklearn.decomposition import PCA 
    pca = PCA(n_components =10,svd_solver ='auto')
    previsores_pca = pca.fit_transform(previsores_escalodados)
    
    print(previsores_pca.shape)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    
    
    
    import pickle
    with open('previsores_pca.pkl','wb') as arquivo:
                pickle.dump(previsores_pca, arquivo)
     

    
    return(0)