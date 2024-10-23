import os
import shutil
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans


image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights="imagenet", include_top=False)


def machine_learning():
    """
    Performs feature extraction and clustering on images with magenta content.

    This function carries out the following steps:
    1. Loads images from the 'output/magenta' directory.
    2. Extracts features from each image using a pre-trained model.
    3. Clusters the features into a specified number of clusters using KMeans.
    4. Saves the clustering results to an Excel file in the 'output/cluster' directory.
    5. Copies clustered images to a designated target directory, renaming them based on their cluster label.

    Returns:
    None
    """
    imdir = os.path.join("output", "magenta")  # DIR containing images
    targetdir = (
        os.path.join("output", "cluster") + "\\"
    )  # DIR to copy clustered images to
    number_clusters = 5

    # Loop over files and get features
    filelist = glob.glob(os.path.join(imdir, "*.png"))
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        try:
            print(f"Status: {i} / {len(filelist)}", end="\r")
            img = image.load_img(imagepath, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            features = np.array(model.predict(img_data))
            featurelist.append(features.flatten())
        except:
            continue

    # Clustering
    kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(
        np.array(featurelist)
    )

    kmeans_result = pd.DataFrame()
    kmeans_result["files"] = filelist
    kmeans_result["labels"] = kmeans.labels_

    kmeans_result.to_excel(os.path.join("output", "cluster", "clusters.xlsx"))

    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    print("\n")
    for i, m in enumerate(kmeans.labels_):
        try:
            print(f"Copy: {i} / {len(kmeans.labels_)}", end="\r")
            shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")
        except:
            continue
