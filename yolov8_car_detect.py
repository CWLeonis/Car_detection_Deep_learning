# -*- coding: utf-8 -*-
"""yolov8_car_detect.ipynb

Original file is located at
    https://colab.research.google.com/drive/18D9aIr84cdr0gR8YxRPU9VWa4icYh025
"""

import zipfile
import requests
import cv2
import matplotlib.pyplot as plt
import glob
import random
import os

!pip install ultralytics

#Download the Dataset
! curl -L "https://app.roboflow.com/ds/2blddBwCO9?key=eBxZH2Jy9L" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

#Visualize Images from the Dataset
import numpy as np
class_names = ['Peugeot 206', 'Peugeot Persia Pars', 'Pride Saipa111', 'Pride Saipa131', 'Renault Tondar 90', 'Saipa Quik']
colors = np.random.uniform(0, 255, size=(len(class_names), 3))


# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels):

    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize the coordinates
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)
        width = xmax - xmin
        height = ymax - ymin

        class_name = class_names[int(labels[box_num])]

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=colors[class_names.index(class_name)],
            thickness=2
        )

        font_scale = min(1,max(3,int(w/500)))
        font_thickness = min(2, max(10,int(w/50)))

        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        # Text width and height
        tw, th = cv2.getTextSize(
            class_name,
            0, fontScale=font_scale, thickness=font_thickness
        )[0]
        p2 = p1[0] + tw, p1[1] + -th - 10
        cv2.rectangle(
            image,
            p1, p2,
            color=colors[class_names.index(class_name)],
            thickness=-1,
        )
        cv2.putText(
            image,
            class_name,
            (xmin+1, ymin-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    return image

# Function to plot images with the bounding boxes.
def plot(image_paths, label_paths, num_samples):
    all_images = []
    all_images.extend(glob.glob(image_paths+'/*.jpg'))
    all_images.extend(glob.glob(image_paths+'/*.JPG'))

    all_images.sort()

    num_images = len(all_images)

    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image_name = all_images[j]
        image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])
        image = cv2.imread(all_images[j])
        with open(os.path.join(label_paths, image_name+'.txt'), 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=1)
    plt.tight_layout()
    plt.show()

# Visualize a few training images.
plot(
    image_paths='/content/train/images',
    label_paths='/content/train/labels',
    num_samples=4,
)

#training for 25 epoch.
EPOCHS = 25

!yolo task=detect mode=train model=yolov8s.pt imgsz=640 data=/content/data.yaml epochs={EPOCHS} batch=16 name=yolov8s_model

#Evaluation on Validation Images
!yolo task=detect mode=val model=runs/detect/yolov8n_v8_50e/weights/best.pt name=yolov8n_eval data=pothole_v8.yaml

#Inference on Images
import glob as glob

def inference(data_path):
    # Directory to store inference results.
    infer_dir_count = len(glob.glob('/content/drive/MyDrive/detect/*'))
    print(f"Current number of inference detection directories: {infer_dir_count}")
    INFER_DIR = f"inference_{infer_dir_count+1}"
    print(INFER_DIR)
    # Inference on images.
    !yolo task=detect \
    mode=predict \
    model=/content/drive/MyDrive/runs/detect/yolov8s_model/weights/best.pt \
    source={data_path} \
    imgsz=640 \
    name={INFER_DIR}
    return INFER_DIR

#Visualize Results
# Plot and visualize images in a 2x2 grid.
def visualize(result_dir, num_samples=1):
    """
    Function accepts a list of images and plots
    them in a 2x2 grid.
    """
    plt.figure(figsize=(20, 12))
    image_names = glob.glob(os.path.join(result_dir, '*.jpg'))
    random.shuffle(image_names)
    for i, image_name in enumerate(image_names):
        image = plt.imread(image_name)
        plt.subplot(2, 2, i+1)
        plt.imshow(image)
        plt.axis('off')
        if i == num_samples-1:
            break
    plt.tight_layout()
    plt.show()

dResult = inference('/content/9.jpg')
visualize('/content/drive/MyDrive/detect/'+dResult)

inference('/content/1.mp4')