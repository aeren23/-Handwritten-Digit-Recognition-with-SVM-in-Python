import tkinter as tk
from tkinter import Canvas
import numpy as np
import cv2
from PIL import Image, ImageGrab,ImageOps
from sklearn.preprocessing import scale
import joblib  # Eğitilmiş modeli yüklemek için
import os
import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas_size = 280  # Daha büyük çizim alanı (daha iyi çözünürlük)
        self.brush_size = 15  # Kalem büyüklüğü
        
        # Tkinter Canvas
        self.canvas = tk.Canvas(root, bg="black", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()
        
        # Çizim için fare olaylarını bağlama
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # Resmi saklamak için numpy array (Siyah arka plan)
        self.image = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

        # Butonlar
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.predict_button = tk.Button(self.button_frame, text="Tahmin Et", command=self.predict)
        self.predict_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(self.button_frame, text="Temizle", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT)

        #modeli entegre et
        model_path = os.path.join(os.getcwd(), "svm_model.pkl")
        self.model = joblib.load(model_path)

    def draw(self, event):
        """Fare hareket ettikçe çizim yap (Beyaz renkte, kenarlara doğru gri efektli)"""
        x, y = event.x, event.y
        r = self.brush_size

        # Çember şeklinde fırça ekleyelim (Merkez beyaz, kenarlar gri)
        for i in range(-r, r):
            for j in range(-r, r):
                if 0 <= x+i < self.canvas_size and 0 <= y+j < self.canvas_size:
                    distance = np.sqrt(i**2 + j**2)
                    if distance <= r:
                        # Daha beyaz merkez için değeri artır
                        intensity = int(255 * (1 - (distance / r)))  # Kenarlarda gri tonlu olacak
                        # Merkezdeki beyaz rengin daha yoğun olmasını sağlamak
                        if distance < r // 2:  # Merkez kısmındaki fırça daha beyaz olacak
                            intensity = 255  # Merkezde beyaz rengi yap
                        self.image[y+j, x+i] = max(self.image[y+j, x+i], intensity)

        # Canvas üzerine çizimi uygula
        self.update_canvas()


    def update_canvas(self):
        """NumPy array'den Tkinter Canvas'a resmi güncelle"""
        img = cv2.resize(self.image, (28, 28), interpolation=cv2.INTER_AREA)  # 28x28'e küçült
        img = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian Blur uygula (daha doğal geçiş)
        img = cv2.resize(img, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_NEAREST)  # Tekrar büyüt
        
        img = Image.fromarray(img)  # PIL formatına çevir
        img = ImageTk.PhotoImage(img)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img  # Referansı kaybetmemek için sakla

    def predict(self):
        """Çizimi 28x28 olarak alıp model tahmini yap"""
        
        # NumPy array üzerinde doğrudan işlemi başlat
        img_resized = cv2.resize(self.image, (28, 28))  # Model için 28x28 boyutuna küçült

        # 28x28 boyutunda resmi kaydet
        #cv2.imwrite('resim_28x28.png', img_resized) 

        img_resized = img_resized.flatten()

        # Ölçeklendirme işlemi
        x_scaled = scale(img_resized)  # Ölçekleme fonksiyonu
        
        # Modelin tahminini yapıyoruz
        prediction = self.model.predict([x_scaled])
        
        print(f"Tahmin Edilen Rakam: {prediction[0]}") 

    def readImage(path):
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=img.flatten()
        img=scale(img)
        return img

    def clear_canvas(self):
        """Canvas'ı temizle"""
        self.image = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        self.update_canvas()



# Tkinter başlat
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()

