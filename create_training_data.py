import os
import cv2 as cv
import random
import matplotlib.pyplot as plt
import numpy as np

GENERATE_FACES = False
GENERATE_NON_FACES = True

# Path to the folder containing the images
TRAIN_IMAGES_PATH = ["./antrenare/barney/", "./antrenare/betty/", "./antrenare/fred/", "./antrenare/wilma/"]
TRAIN_IMAGES_LABELS = ["./antrenare/barney_annotations.txt", "./antrenare/betty_annotations.txt",
                       "./antrenare/fred_annotations.txt", "./antrenare/wilma_annotations.txt"]
TRAIN_IMAGES = []
TRAIN_LABELS = {}

# Sliding window size
sliding_window_size = (36, 36)

# Parameters for the sliding window
image_resize_factor = 1
random_window_max_tries = 5

low_skin_color = (5, 45, 105)
high_skin_color = (25, 255, 255)

def show_hsv_colors(high_hsv, low_hsv):
    # Convert the HSV values to BGR color space
    # high_bgr = cv.cvtColor(np.uint8([[high_hsv]]), cv.COLOR_HSV2BGR)[0][0]
    # low_bgr = cv.cvtColor(np.uint8([[low_hsv]]), cv.COLOR_HSV2BGR)[0][0]

    # Create 256x256 images of solid color for each HSV value
    high_color_img = np.full((256, 256, 3), high_hsv, np.uint8)
    low_color_img = np.full((256, 256, 3), low_hsv, np.uint8)

    # Display the images
    cv.imshow('High Skin Color', high_color_img)
    cv.imshow('Low Skin Color', low_color_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# show_hsv_colors(high_skin_color, low_skin_color)

train_data_images_with_faces = []
train_data_images_without_faces = []

barney = []
betty = []
fred = []
wilma = []


def show_image(title, imagee):
    imagee = cv.resize(imagee,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,imagee)
    cv.waitKey(0)
    cv.destroyAllWindows()
    plt.imshow(imagee)
    plt.show()

# To get the face from the image
def get_image_from_label(img, annotation):
    return img[int(annotation[1]):int(annotation[3]), int(annotation[0]):int(annotation[2])]

def get_image_and_add_5_pixels(img, annotation):
    y1 = max(int(annotation[1]) - 5, 0)
    y2 = min(int(annotation[3]) + 5, img.shape[0])
    x1 = max(int(annotation[0]) - 5, 0)
    x2 = min(int(annotation[2]) + 5, img.shape[1])
    return img[y1:y2, x1:x2]

def check_overlapping(l1, r1, l2, r2):
    return l1[0] < r2[0] and r1[0] > l2[0] and l1[1] < r2[1] and r1[1] > l2[1]

# Read images
for folder in TRAIN_IMAGES_PATH:
    for image in os.listdir(folder):
        TRAIN_IMAGES.append(os.path.join(folder + image))

# Read labels
for label in TRAIN_IMAGES_LABELS:
    with open(label) as f:
        for line in f:
            data = line.split()
            key = label.split("/")[-1].split("_")[0] + "_" + data[0]
            x1, y1, x2, y2, name = int(data[1]), int(data[2]), int(data[3]), int(data[4]), data[5]

            if key in TRAIN_LABELS.keys():
                dict_value = TRAIN_LABELS.get(key)
                dict_value.append((x1, y1, x2, y2, name))
                TRAIN_LABELS[key] = dict_value
            else:
                TRAIN_LABELS[key] = [(x1, y1, x2, y2, name)]

for image_path in TRAIN_IMAGES:
    # Load labels
    image_labels = TRAIN_LABELS[image_path.split("/")[-2] + "_" + image_path.split("/")[-1]]
    image = cv.imread(image_path)

    # Append face from labels
    if GENERATE_FACES:
        for image_label in image_labels:
            face = get_image_from_label(image, image_label)
            face_resized = cv.resize(face, (sliding_window_size[0] * image_resize_factor, sliding_window_size[1] * image_resize_factor))
            train_data_images_with_faces.append(face_resized)
            face_with_5_pixels = get_image_and_add_5_pixels(image, image_label)
            face_with_5_pixels_resized = cv.resize(face_with_5_pixels, (sliding_window_size[0] * image_resize_factor, sliding_window_size[1] * image_resize_factor))
            train_data_images_with_faces.append(face_with_5_pixels_resized)

    if GENERATE_NON_FACES:
        for i in range(20):
            for j in range(5):
                # Select random coords
                rand_x, rand_y = random.randint(0, image.shape[1] - int(sliding_window_size[0] * (1.25**j) + 1)), random.randint(0, image.shape[0] - int(sliding_window_size[1] * (1.25**j) + 1))

                # Check if coords are overlapping with a face
                overlapping = False
                for image_label in image_labels:
                    if check_overlapping((rand_x, rand_y), (rand_x + int(sliding_window_size[0] * (1.25**j)), rand_y + int(sliding_window_size[1] * (1.25**j))), (image_label[0], image_label[1]), (image_label[2], image_label[3])):
                        overlapping = True
                        break

                if not overlapping:
                    negative_example = image[rand_y:rand_y + int(sliding_window_size[1] * (1.25 ** j) + 1),
                                       rand_x:rand_x + int(sliding_window_size[0] * (1.25 ** j) + 1)]
                    patch_hsv = cv.cvtColor(negative_example, cv.COLOR_BGR2HSV)
                    skin_patch = cv.inRange(patch_hsv, low_skin_color, high_skin_color)
                    # cv.imshow("Skin Patch", skin_patch)
                    # cv.waitKey(0)
                    # print(skin_patch.mean())
                    if skin_patch.mean() >= 70: #initial era 60 tine minte radu
                        non_face = image[rand_y:rand_y + int(sliding_window_size[1] * (1.25**j) + 1), rand_x:rand_x + int(sliding_window_size[0] * (1.25**j) + 1)]
                        non_face = cv.resize(non_face, (sliding_window_size[0] * image_resize_factor, sliding_window_size[1] * image_resize_factor))
                        train_data_images_without_faces.append(non_face)


if GENERATE_FACES:
    for i, image in enumerate(train_data_images_with_faces):
        cv.imwrite("./positive_examples/" + str(i) + ".jpg", image)

if GENERATE_NON_FACES:
    print(len(train_data_images_without_faces))
    for i, image in enumerate(train_data_images_without_faces):
        cv.imwrite("./negative_examples/" + str(i) + ".jpg", image)