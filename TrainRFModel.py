from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def trainRFModel(paramerter,maskImgArr):

    #Spliting Data
    #Define the dependent variable that needs to be predicted (labels)
    Y = paramerter["Labels"].values

    #Define the independent variables
    X = paramerter.drop(labels = ["Labels"], axis=1)

    # Instantiate model with n number of decision trees
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)

    # Train the model on training data
    model.fit(X,Y)

    return(model)
