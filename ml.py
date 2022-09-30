# https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/
# https://www.geeksforgeeks.org/python-image-classification-using-keras/


# explain ML
# https://www.analyticsvidhya.com/blog/2020/03/6-python-libraries-interpret-machine-learning-models/

# Use mahotas to detect features from templates (e.g. head, tail)?

# Use templates to get features / crops?
# https://www.geeksforgeeks.org/template-matching-using-opencv-in-python/


# This one seems promising
# https://datascience.stackexchange.com/questions/63434/unsupervised-image-classification

# Imports
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = True 
model = VGG16(weights='imagenet', include_top=False)
import pandas as pd

# Variables
imdir = os.path.join('input', 'crop') # DIR containing images
targetdir = os.path.join('input', 'cluster') + '\\' # DIR to copy clustered images to
number_clusters = 5

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.tif'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    try:
        print("    Status: %s / %s" %(i, len(filelist)), end="\r")
        img = image.load_img(imagepath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())
    except:
        continue

# Clustering
kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))


kmeans_result = pd.DataFrame()
kmeans_result['files'] = filelist
kmeans_result['lables'] = kmeans.labels_

kmeans_result.to_excel(os.path.join('input', 'cluster', 'clusters.xlsx'))

# Copy images renamed by cluster 
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    try:
        print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
        shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")
    except:
        continue


# TODO: Flip crops into the same orientation... How to determine if fish is left/right facing?
# Potentially using a histogram of non-zero values in thresh?

# TODO: Only run script on magenta > Just comparing bones? > This also ignores black crops