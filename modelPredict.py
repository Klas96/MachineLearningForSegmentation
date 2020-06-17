from DataToParmeters import loadDataForModel
from TrainRFModel import trainRFModel

def modelPredict(model,floImg,optImg,modelParam):
    result = model.predict(modelParam)
    predictImg = result.reshape((floImg.shape))
    
    return(predictImg)
