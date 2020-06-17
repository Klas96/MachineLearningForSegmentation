from DataToParmeters import loadDataForModel
from TrainRFModel import trainRFModel
from TrainSVModel import trainSVModel
import cv2
from modelPredict import modelPredict
from matplotlib import pyplot as plt
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
import pickle

numTrainImg = 7

optImgArr = []
floImgArr = []
maskImgArr = []

for imgNum in range(numTrainImg):
    optImgPath = 'YeastCell/Train/' + str(imgNum) + 'C0Z3.png'
    floImgPath = 'YeastCell/Train/' + str(imgNum) + 'C1Z1.png'
    maskImgPath = 'YeastCell/Train/' + str(imgNum) + 'Mask.png'
    #Load as GrayScale
    optImg = cv2.imread(optImgPath,0)
    floImg = cv2.imread(floImgPath,0)
    maskImg = cv2.imread(maskImgPath,0)
    optImgArr.append(optImg)
    floImgArr.append(floImg)
    maskImgArr.append(maskImg)

df = loadDataForModel(optImgArr,floImgArr,maskImgArr)
df = pd.read_csv('YeastCell/Train/modelTrain.csv')

rfModel = trainRFModel(df,maskImgArr)

#Save Model
filename = "YeastCellRFModel"
pickle.dump(rfModel, open(filename, 'wb'))

###############
##Load images##
###############

optImg = cv2.imread('YeastCell/Test/TestC0Z3.jpg')
floImg = cv2.imread('YeastCell/Test/TestC1Z2.jpg')
optImg = cv2.cvtColor(optImg, cv2.COLOR_BGR2GRAY)
floImg = cv2.cvtColor(floImg, cv2.COLOR_BGR2GRAY)

df = loadDataForModel([optImg],[floImg])

print("Head:")
print(df.head())

predictImg = modelPredict(rfModel,floImg,optImg,df)

plt.imshow(optImg)
#plt.show()
plt.imshow(predictImg)
plt.show()
