import cv2
#import matplotlib.pyplot as plt
import h5py
import numpy as np
import mahotas
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import hsv_to_rgb
import os
h5_train_data          = '/home/pi/Downloads/Plant-disease-detection/output/train_data.h5'
h5_train_labels        = '/home/pi/Downloads/Plant-disease-detection/output/train_labels.h5'
images_per_class=800
fixed_size=tuple((500,500))
bins=8
train_path="/home/pi/Documents/Plant-Disease-detection/image_classification/dataset/train"

#img=cv2.imread("/home/pi/Downloads/b7350a31-68e1.jpeg")
def rgb_bgr(img):
    rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return rgb_img
def hsv_img(img):
    hsv_img1=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    return hsv_img1
def img_seg(rgb_img,hsv_img):
    lower_green=np.array([25,0,20])
    upper_green=np.array([100,255,255])
    healthy_mask=cv2.inRange(hsv_img,lower_green,upper_green)
    result=cv2.bitwise_and(rgb_img,rgb_img,mask=healthy_mask)
    lower_brown=np.array([10,0,10])
    upper_brown=np.array([30,255,255])
    disease_mask=cv2.inRange(hsv_img,lower_brown,upper_brown)
    disease_result=cv2.bitwise_and(rgb_img,rgb_img,mask=disease_mask)
    final_mask=healthy_mask+disease_mask
    final_result=cv2.bitwise_and(rgb_img,rgb_img,mask=final_mask)
    #plt.subplot(1,2,1)
    #plt.imshow(img,cmap="gray")
    #plt.show()
    #plt.subplot(1,2,2)
    #plt.imshow(disease_mask)
    #plt.show()
    return final_result
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
train_labels = os.listdir(train_path)
# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels          = []

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"

        # read the image and resize it to a fixed-size
        image1 = cv2.imread(file)
        image = cv2.resize(image1, fixed_size)

        
        # Running Function Bit By Bit
        
        RGB_BGR       = rgb_bgr(image)
        BGR_HSV       = hsv_img(RGB_BGR)
        IMG_SEGMENT   = img_seg(RGB_BGR,BGR_HSV)

        # Call for Global Fetaure Descriptors
        
        fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
        fv_haralick   = fd_haralick(IMG_SEGMENT)
        fv_histogram  = fd_histogram(IMG_SEGMENT)
        
        # Concatenate 
        
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        
        

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))
print("[STATUS] training Labels {}".format(np.array(labels).shape))
# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")
# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

h5f_data = h5py.File(h5_train_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_train_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()
