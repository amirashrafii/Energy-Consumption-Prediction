for K in range(20):
    K = K+1
# K=7
    #for i, weights in enumerate(['uniform', 'distance']):
    model = neighbors.KNeighborsRegressor(n_neighbors=K)
    model.fit(x_train, y_train) #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error, 'and R2 score is:', model.score(x_test,y_test))