#Destek vektör makineleri ile el yazısı karakter tanıma
#28x28 boyutundaki resimleri 784 parametre ile ifade edebiliriz.
#Bu resimlerin 10 farklı sınıfı vardır. Bu sınıflar 0-9 arasındaki rakamlardır.

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import cv2

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#Veri setini yükle
numbers=pd.read_csv("train.csv")

one=numbers.iloc[2,1:]
one=one.values.reshape(28,28)
plt.imshow(one)
plt.title("Resimdeki Rakam: 1")
plt.show()

#(x) parametrelerini ve (y) etiket değişkenlerini ayrıştır
x=numbers.drop(["label"],axis=1)
y=numbers["label"]

#Veri setini eğitim ve test setlerine ayır
#Eğitim seti: 80%, Test seti: 20%

X=scale(x)
X_train, X_test,y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=101)

model_rbf = SVC(kernel='rbf',gamma=0.001,C=10)
model_rbf.fit(X_train,y_train)

#Eğitim seti doğruluğu: sckit-learn kütüphanesindeki accuracy_score fonksiyonu ile hesaplanır
pred=model_rbf.predict(X_test)
print("Doğruluk Oranı: ",metrics.accuracy_score(y_true=y_test,y_pred=pred),"\n")

#karmaşıklık matrisi
print(metrics.confusion_matrix(y_true=y_test,y_pred=pred),"\n")

def readImage(path):
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img=img.flatten()
    return img



#joblib.dump(model_rbf, 'svm_modelrbf.pkl')

predict_data=readImage("img_0.png")
predict_data=model_rbf.predict([predict_data])
print("Tahmin Edilen Rakam: ",str(predict_data))