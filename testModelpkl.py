import cv2
from sklearn.preprocessing import scale
import joblib 
from PIL import Image
import numpy as np 
from sklearn.preprocessing import scale

#Bu dosya eğitilen modeli görsellerden test etmek için oluşturuldu

def readImage(path):
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img=img.flatten()
    img=scale(img)
    return img


model=joblib.load('svm_model.pkl')

predict_data=readImage("resim_28x28.png")
predict_data=model.predict([predict_data])
print("Tahmin Edilen Rakam: ",str(predict_data))
