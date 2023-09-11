import sys
#sys.path.append('../input/imutils-054/imutils-0.5.4')

import imutils
import os
import cv2
import datetime
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

import pickle

!curl -L "YOUR DATASET LINK HERE"  #****************
#Put your class names in classes list below***********
classes = ['Pride Saipa131', 'pars', 'pezho206', 'pride', 'quick', 'tondar'] 

def something(ann_path , images_path):
    data = []
    labels = []
    bboxes = []
    imagePaths = []
    #ann_path_new = ann_path + cl + "_train.csv"
    print(ann_path)
    rows = open(ann_path).read().strip().split("\n")
    rows = rows[1:]
    # loop over the rows
    for idx, row in enumerate(rows): #row = col
        # break the row into the
        # filename and bounding box coordinates
        row = row.split(",")
        filename = row[0]
        #filename = filename.split(".")[0]
        #filename = filename.split("_")[-1]
        #filename = filename + ".jpg"
        coords = row[4:]
        coords = [int(c) for c in coords]
        label = row[3]
        image_path = os.path.sep.join([images_path, filename])
        image = cv2.imread(image_path)
        (h, w) = image.shape[:2]


        Xmin = float(coords[0]) / w
        Ymin = float(coords[1]) / h
        Xmax = float(coords[2]) / w
        Ymax = float(coords[3]) / h

        # load the image
        image = load_img(image_path, target_size=(224, 224))  #color_mode="grayscale"
        image = img_to_array(image)

        data.append(image)
        labels.append(label)
        bboxes.append((Xmin, Ymin, Xmax, Ymax))
        imagePaths.append(image_path)
    return data, labels, bboxes, imagePaths

'''
# show the output image
imgplot = plt.imshow(image.astype('uint8'))
plt.show()
'''

ann_path = "/train/_annotations.csv"
images_path = "/train"
data_train = []
labels_train = []
bboxes_train = []
imagePaths_train = []
data_train, labels_train, bboxes_train, imagePaths_train  = something(ann_path, images_path)

# convert from the range [0, 255] to [0, 1]
data_train = np.array(data_train, dtype="float32") / 255.0
# convert to numpy array
labels_train = np.array(labels_train)
bboxes_train = np.array(bboxes_train, dtype="float32")
imagePaths_train = np.array(imagePaths_train)
#*************************************************************
ann_path = "/valid/_annotations.csv"
images_path = "/valid"
data_valid = []
labels_valid = []
bboxes_valid = []
imagePaths_valid = []
data_valid, labels_valid, bboxes_valid, imagePaths_valid  = something(ann_path, images_path)

data_valid = np.array(data_valid, dtype="float32") / 255.0
labels_valid = np.array(labels_valid)
bboxes_valid = np.array(bboxes_valid, dtype="float32")
imagePaths_valid = np.array(imagePaths_valid)
#************************************************************
ann_path = "/test/_annotations.csv"
images_path = "/test"
data_test = []
labels_test = []
bboxes_test = []
imagePaths_test = []
data_test, labels_test, bboxes_test, imagePaths_test  = something(ann_path, images_path)

data_test = np.array(data_test, dtype="float32") / 255.0
labels_test = np.array(labels_test)
bboxes_test = np.array(bboxes_test, dtype="float32")
imagePaths_test = np.array(imagePaths_test)

# one-hot encoding on the labels
lb = LabelBinarizer()
labels_train = lb.fit_transform(labels_train)

labels_valid = lb.fit_transform(labels_valid)

labels_test = lb.fit_transform(labels_test)

if len(lb.classes_) == 2:
    print("two classes")
    labels = to_categorical(labels)

trainImages , validImages = data_train , data_valid
trainLabels, validLabels = labels_train , labels_valid
trainBBoxes, validBBoxes = bboxes_train , bboxes_valid
trainPaths, validPaths = imagePaths_train, imagePaths_valid

testImages = data_test
testLabels = labels_test
testBBoxes = bboxes_test
testPaths = imagePaths_test

f = open("testing_multiclass.txt", "w")
f.write("\n".join(validPaths))
f.close()

vgg = VGG16(weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(224, 224,3)))

# freeze all layers of VGG in order not to train them
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer
# header to output the predicted
# bounding box coordinates

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

# construct a second fully-connected
# layer header to predict
# the class label

softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

model = Model(
    inputs=vgg.input,
    outputs=(bboxHead, softmaxHead))

INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32

losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}

lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}

validTargets = {
    "class_label": validLabels,
    "bounding_box": validBBoxes
}

testTargets = {
    "class_label": testLabels,
    "bounding_box": testBBoxes
}

opt = Adam(INIT_LR)

model.compile(loss=losses,
              optimizer=opt,
              metrics=["accuracy"],
              loss_weights=lossWeights)

print(model.summary())

H = model.fit(
    trainImages, trainTargets,
    validation_data=(validImages, validTargets),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=2)

model.save("model_car_detect", save_format="h5")

f = open("lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

lossNames = ["loss",
             "class_label_loss",
             "bounding_box_loss"]

N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(17, 25))

# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()

# create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure(figsize=(17, 10))

plt.plot(N, H.history["class_label_accuracy"],
         label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"],
         label="val_class_label_acc")

plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

# save the accuracies plot
plotPath = os.path.sep.join(["accs.png"])
plt.savefig(plotPath)

path = "testing_multiclass.txt"
filenames = open(path).read().strip().split("\n")
imagePaths = []

for f in filenames:
    imagePaths.append(f)

imagePath = '*** PATH TO JPG FOR THE TEST***'
model = load_model("/model_car_detect")
lb = pickle.loads(open("/lb.pickle", "rb").read())

# load the input image
image = load_img(imagePath, target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)
# predict coordinates and classes
(boxPreds, labelPreds) = model.predict(image)
(startX, startY, endX, endY) = boxPreds[0]

i = np.argmax(labelPreds, axis=1)
label = lb.classes_[i][0]

# load the input image (in OpenCV format)
image = cv2.imread(imagePath)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# scale the predicted bounding box
# coordinates based on the image
# dimensions
startX = int(startX * w)
startY = int(startY * h)
endX = int(endX * w)
endY = int(endY * h)


# draw the predicted bounding
# box and class label on the image
y = startY - 10 if startY - 10 > 10 else startY + 10

cv2.putText(image,
        label,
        (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2)

cv2.rectangle(image,
              (startX, startY),
              (endX, endY),
              (0, 255, 0),
              2)


# show the output image
imgplot = plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype('uint8'))
plt.axis('off')
plt.show()
