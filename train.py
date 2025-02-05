#Support Vector Machine kullanarak modelimizi eğiteceğiz

#Destek vektör makineleri ile el yazısı karakter tanıma
#28x28 boyutundaki resimleri 784 parametre ile ifade edebiliriz.
#Bu resimlerin 10 farklı sınıfı vardır. Bu sınıflar 0-9 arasındaki rakamlardır.

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC


#Veri setini yükle
numbers=pd.read_csv("train.csv")

#(x) parametrelerini ve (y) etiket değişkenlerini ayrıştır
x=numbers.drop(["label"],axis=1)
y=numbers["label"]

# Feature scaling işlemi
X_scaled = scale(x)

# Modeli eğit
model= SVC(kernel='rbf', gamma=0.001, C=10)
model.fit(X_scaled, y)

print("Model eğitimi başarılı!")

# Modeli eğittikten sonra kaydet
joblib.dump(model, 'svm_model.pkl')


# # Modeli yükle
# loaded_model = joblib.load('svm_model.pkl')

# # Test verisi ile tahmin yap
# predictions = loaded_model.predict(X_test_scaled)  # Burada X_test_scaled'e ihtiyacın olacak
