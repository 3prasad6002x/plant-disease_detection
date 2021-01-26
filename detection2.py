def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
import cv2
from picamera import PiCamera
from time import sleep
import h5py
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import mahotas
from matplotlib.colors import hsv_to_rgb
seed = 9
h5_train_data          = '/home/pi/Downloads/Plant-disease-detection/output/train_data.h5'
h5_train_labels        = '/home/pi/Downloads/Plant-disease-detection/output/train_labels.h5'
fixed_size=tuple((800,800))
bins=8
num_trees=100

h5f_data  = h5py.File(h5_train_data, 'r')
h5f_label = h5py.File(h5_train_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

import cv2
camera=PiCamera()
camera.resolution=(1024,768)
camera.start_preview()
camera.brightness=40
sleep(20)
camera.capture("/home/pi/Downloads/plant-disease-detection21.jpg")
camera.stop_preview()
image=cv2.imread("/home/pi/Downloads/plant-disease-detection21.jpg")
image = cv2.resize(image, fixed_size)


rgb_img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
hsv_img1=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

lower_green=np.array([25,0,20])
upper_green=np.array([100,255,255])
healthy_mask=cv2.inRange(hsv_img1,lower_green,upper_green)
result=cv2.bitwise_and(rgb_img,rgb_img,mask=healthy_mask)
lower_brown=np.array([10,0,10])
upper_brown=np.array([30,255,255])
disease_mask=cv2.inRange(hsv_img1,lower_brown,upper_brown)
disease_result=cv2.bitwise_and(rgb_img,rgb_img,mask=disease_mask)
final_mask=healthy_mask+disease_mask
final_result=cv2.bitwise_and(rgb_img,rgb_img,mask=final_mask)
plt.subplot(3,1,1)
plt.imshow(rgb_img)
plt.title("original")
plt.subplot(3,1,2)
plt.imshow(final_mask,cmap="gray")
plt.title("processing")
plt.subplot(3,1,3)
plt.imshow(final_result)
plt.title("final")
plt.show()


fv_hu_moments = fd_hu_moments(final_result)
fv_haralick   = fd_haralick(final_result)
fv_histogram  = fd_histogram(final_result)      
global_feature1 = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
#print(global_feature1)
#global_feature1=global_feature1.reshape(-1,1)
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0, 1))
rescaled_features1 = scaler.fit_transform(global_feature1.reshape(1,-1))
print("[STATUS] feature vector normalized...")

print("[STATUS] training started...")
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(np.array(global_features),np.array(global_labels),test_size=0.2,random_state=9)
from sklearn.ensemble import RandomForestClassifier
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
clf.fit(X_train,Y_train)
y_predict1=clf.predict(X_test)
y_predict=clf.predict(global_feature1.reshape(1,-1))
print(y_predict)
from sklearn.metrics import accuracy_score,precision_score,recall_score
accuracy=accuracy_score(Y_test,y_predict1)
print("accuracy:", accuracy)
precision=precision_score(Y_test,y_predict1)
print("precision:", precision)
recall=recall_score(Y_test,y_predict1)
print("recall:",recall)
if(y_predict==np.array(0)):
    print("INFECTED WITH DISEASE")
else:
    print("HEALTHY PLANT")
