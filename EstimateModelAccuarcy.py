from sklearn import metrics

def estimateModelAccuarcy(model,tarinData):
    prediction_test_train = model.predict(X_train)

    #Test prediction on testing data.
    prediction_test = model.predict(X_test)
    #First check the accuracy on training data. This will be higher than test data prediction accuracy.
    print("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
    #Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
    print("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

    #This part commented out for SVM testing. Uncomment for random forest.
    #One amazing feature of Random forest is that it provides us info on feature importances
    # Get numerical feature importances
    #importances = list(model.feature_importances_)

    #Let us print them into a nice format.

    feature_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
    print(feature_imp)
