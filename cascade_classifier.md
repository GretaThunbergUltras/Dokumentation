# OpenCV Haar Cascade

Systemanforderung:
- OpenCV muss vom Quellcode kompiliert werden (pip3 installopencv-python reicht nicht)
- Bisher hab ich nur auf Ubuntu getestet
- Desto besser der Prozessor ist, desto schneller geht das Training
- Min. 3-4GB Arbeitsspeicher

Notwendige Ordnerstruktur:
- Ordner „data“ zum Speichern des Cascade-Files
- Ordner „info“ zum Speichern der positiven Bilder
- Ordner „neg“ zum Speichern der negativen Bilder
- positive_bild.jpg (Bild mit dem trainiert werden soll, auf max. 50x50px zurechtgeschnitten; idealerweise 50x50px falls quadratisch, andernfalls lange Seite max. 50px)

## CascadeClassifier erstellen

1. Bilderdatensatz auf image-net.org aussuchen
2. Python-Script um Bilder herunterzuladen und im „neg“ Ordner zu speichern, auf 100x100px zu skalieren und in Graustufen umzuwandeln

```
import cv2
import numpy as np
import urllib.request
import os
 
def store_raw_images():
  neg_images_link = "http://www.imagenet.org/api/text/imagenet.synset.geturls?wnid=n00007846"
  neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
 
if not os.path.exists('neg'):
  os.makedirs('neg')
  
pic_num=980

for i in neg_image_urls.split('\n'):
  try:
    print(i)
    urllib.request.urlretrieve(i,"neg/"+str(pic_num)+".jpg")
    img = cv2.imread("neg/"+str(pic_num)+".jpg", cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img,(100,100))
    cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
    pic_num+=1
  except Exception as e:
    print(str(e))
```

3.	Python-Script um bg.txt Datei zu erstellen, die für späteres Training notwendig ist

```
def create_pos_neg():
  for file_type in ['neg']:
    for img in os.listdir(file_type):
      if file_type == 'neg':
        line = file_type+"/"+img+"\n"
        with open("bg.text","a") as f:
          f.write(line)
```

4.	Danach muss der Cascade-Ordner im Terminal geöffnet werden

```
opencv_createsamples-img palette_front_50_47.jpg -bgbg.text-info info/info.lst-pngoutputinfo-maxxangle0.5-maxyangle0.5-maxzangle0.5-num1750 
#-num muss kleiner als die Anzahl der negativen Bilder
```

5.	Erstellung des Vektorfiles für die positiven Bilder

```
openv_createsamples-info info/info.lst-num1750-w 48-h 48-vecpositives.vec
#-num muss gleich groß sein wie im vorherigen Schritt
#-w und -h gibt Größe des zu trainierenden Classifiers an, desto größer #desto länger dauert das Training und desto mehr Rechenleistung ist nötig
#48x48 bereits ziemlich rechenintensiv
```

6. Training

```
opencv_traincascade-data data-vecpositives.vec-bgbg.text-numPos1600-numNeg800-numStages15-w 48-h 48
#-numPos darf nicht größer sein als in den vorherigen Schritten festgelegt
#Verhältnis von numPos zu numNeg sollte in der Regel 2:1 sein, deswegen -#numNeg 800
#-w und -h muss genauso groß sein wie im vorherigen Schritt festgelegt
#-numStages gibt die Anzahl der zu trainierenden Stages an, desto mehr #Stages, desto besser wird die Cascade (kann aber auch übertrainiert #werden). Jedoch dauert das Training auch um einiges länger
```

7. Fertige Cascade-Datei. Im Ordner data sind nun die einzelnen .xml-Dateien der Stages abgespeichert. "cascade.xml" ist der fertige Cascade-Classifier.

## Classifier verwenden

```
import numpy as np
import cv2
 
def nothing():
  pass
 
cascade = cv2.CascadeClassifier('cascade.xml')
cap = cv2.VideoCapture(0)
 
#Trackbars fürdetectMultiScale parameter (scale_factor und min_neighbours)
cv2.namedWindow("Trackbars")
cv2.createTrackbar("scale","Trackbars",11,20, nothing)
cv2.createTrackbar("min","Trackbars",3,6, nothing)
cv2.createTrackbar("Size","Trackbars",100,200, nothing)
 
while 1:
  ret, img = cap.read()
  resize_val = cv2.getTrackbarPos("Size","Trackbars")
  img = cv2.resize(img,(int(img.shape[1]*resize_val/100),int(img.shape[0]*resize_val/100)))
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
  objects=cascade.detectMultiScale(gray,cv2.getTrackbarPos("scale","Trackbars")/10,	cv2.getTrackbarPos("min","Trackbars"))
 
  for (x, y, w, h) in objects:
    print("Object detected")
    print("X: {:d}, Y: {:d}, W: {:d}, H: {:d}".format(x, y, w, h))
    cv2.rectangle(img,(x, y),(x + w, y + h),(255,255,0),2)
    break

  cv2.imshow('img',img)
  k = cv2.waitKey(30)&0xff
  if k == 27:
    break
 
cap.release()
cv2.destroyAllWindows()
```
