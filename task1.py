import os
import pickle
import timeit

import cv2 as cv
import numpy as np
from skimage.feature import hog
from sklearn import svm

from Visualize import show_detections_with_ground_truth

SAVE_MODEL = True
LOAD_MODEL = False
TRAIN_MODEL = True
LOAD_IMAGES = True
SAVE_FILES = True

POSITIVE_EXAMPLES_PATH = "./positive_examples/"
NEGATIVE_EXAMPLES_PATH = "./negative_examples/"
VALIDATION_PATH = "./validare/validare/"
GROUND_TRUTH_PATH = "./validare/task1_gt_validare.txt"

# Model params
sliding_window_size = (8, 8)
sliding_window_step_size = 1

# Training params
train_window_size = (36, 36)
hog_pixels_per_cell = (6, 6)

images = []
labels = {}

low_skin_color = (5, 45, 105)
high_skin_color = (25, 255, 255)

def check_overlapping(l1, r1, l2, r2):
    return l1[0] < r2[0] and r1[0] > l2[0] and l1[1] < r2[1] and r1[1] > l2[1]

def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

def non_maximal_suppression(image_detections, image_scores, image_size):

    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]

    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]

    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = 0.3

    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True:
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j]:
                    if intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:
                        is_maximal[j] = False
                    else:
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                            is_maximal[j] = False

    return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

def main():
    train_data_images_with_faces = []
    train_data_images_without_faces = []

    if LOAD_IMAGES:
        print("Loading images..")

        # Load positive examples files
        positive_descriptors_path = "./descriptors/positive/positive.npy"

        if os.path.exists(positive_descriptors_path):
            images_with_faces_train_data = np.load(positive_descriptors_path)
        else:
            for file in os.listdir(POSITIVE_EXAMPLES_PATH):
                # Load and save image
                image = cv.imread(POSITIVE_EXAMPLES_PATH + file, cv.IMREAD_GRAYSCALE)
                face_hog_image = hog(image, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2), feature_vector=True, orientations = 18)
                train_data_images_with_faces.append(face_hog_image)

                # Augmentation
                face_hog_image = hog(np.fliplr(image), pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                                     feature_vector=True, orientations = 18)
                train_data_images_with_faces.append(face_hog_image)

                image_cropped_top = image[5:, :]
                resized = cv.resize(image_cropped_top, (36, 36))
                face_hog_image = hog(resized, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                                     feature_vector=True, orientations = 18)
                train_data_images_with_faces.append(face_hog_image)

                image_cropped_bottom = image[:image.shape[0] - 5, :]
                resized = cv.resize(image_cropped_bottom, (36, 36))
                face_hog_image = hog(resized, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                                     feature_vector=True, orientations = 18)
                train_data_images_with_faces.append(face_hog_image)

                image_cropped_left = image[:, 5:]
                resized = cv.resize(image_cropped_left, (36, 36))
                face_hog_image = hog(resized, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                                     feature_vector=True, orientations = 18)
                train_data_images_with_faces.append(face_hog_image)

                image_cropped_right = image[:, :image.shape[1] - 5]
                resized = cv.resize(image_cropped_right, (36, 36))
                face_hog_image = hog(resized, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                                     feature_vector=True, orientations = 18)
                train_data_images_with_faces.append(face_hog_image)

            images_with_faces_train_data = np.array(train_data_images_with_faces)
            np.save(positive_descriptors_path, images_with_faces_train_data)

        # Load negative examples files
        negative_descriptors_path = "./descriptors/negative/negative.npy"
        if os.path.exists(negative_descriptors_path):
            images_without_faces_train_data = np.load(negative_descriptors_path)
        else:
            for file in os.listdir(NEGATIVE_EXAMPLES_PATH):
                # Load and save image
                image = cv.imread(NEGATIVE_EXAMPLES_PATH + file, cv.IMREAD_GRAYSCALE)
                non_face_hog_image = hog(image, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                                         feature_vector=True, orientations = 18)
                train_data_images_without_faces.append(non_face_hog_image)

                # Flip image to do small augmentation and append it
                non_face_hog_image = hog(np.fliplr(image), pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                                         feature_vector=True, orientations = 18)
                train_data_images_without_faces.append(non_face_hog_image)

            images_without_faces_train_data = np.array(train_data_images_without_faces)
            np.save(negative_descriptors_path, images_without_faces_train_data)

        print("Images loaded!")

        # Build classifier labels
        print(f"Faces: {len(images_with_faces_train_data)}, Non faces: {len(images_without_faces_train_data)}")
        train_y = np.concatenate(
            (np.ones(images_with_faces_train_data.shape[0]), np.zeros(images_without_faces_train_data.shape[0])))

        # Combine training data
        train_x = np.concatenate(
            (np.squeeze(images_with_faces_train_data), np.squeeze(images_without_faces_train_data)), axis=0)

    # Define Linear SVC
    classifier = svm.LinearSVC(C=1)

    if TRAIN_MODEL:
        print("Training SVM..")
        classifier.fit(train_x, train_y)
        print("SVM Trained!")

        # Run detector on negative examples to find strong negatives (hard negative mining)
        strong_negatives = []
        for file in os.listdir(NEGATIVE_EXAMPLES_PATH):
            # Load and process image
            image = cv.imread(NEGATIVE_EXAMPLES_PATH + file, cv.IMREAD_GRAYSCALE)
            image_resized = cv.resize(image, train_window_size)
            hog_descriptor = hog(image_resized, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                                 feature_vector=True, orientations = 18)

            # Predict using the classifier
            score = classifier.decision_function([hog_descriptor])

            # If the detector fires, this is a strong negative
            if score > 0:
                strong_negatives.append(hog_descriptor)

        # If you have found any strong negatives, retrain the classifier
        if strong_negatives:
            strong_negatives = np.array(strong_negatives)
            # Add strong negatives to the training data
            train_x = np.concatenate((train_x, strong_negatives), axis=0)
            # Labels for strong negatives are 0
            train_y = np.concatenate((train_y, np.zeros(len(strong_negatives))), axis=0)

            # Retrain the classifier with the updated dataset
            print("Retraining SVM with strong negatives..")
            classifier.fit(train_x, train_y)
            print("SVM Retrained!")

    # Save SVM model
    if SAVE_MODEL:
        print("Saving SVM..")
        filename = 'finalized_model_task1.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        print("SVM Saved!")

    # Load SVM model
    if LOAD_MODEL:
        print("Loading SVM..")
        filename = 'finalized_model_task1.sav'
        classifier = pickle.load(open(filename, 'rb'))
        print("SVM Loaded!")

    final_detections = []
    final_file_paths = []
    final_scores = []
    length_of_files = len(os.listdir(VALIDATION_PATH))

    for file_no, file in enumerate(os.listdir(VALIDATION_PATH)):
        start_time = timeit.default_timer()

        # Sliding window for image
        loaded_image = cv.imread('./validare/validare/' + file)
        loaded_image_hsv = cv.cvtColor(loaded_image, cv.COLOR_BGR2HSV)
        loaded_image_hsv_skin = cv.inRange(loaded_image_hsv, low_skin_color, high_skin_color)

        detections = []
        scores = []

        scale_x = 0.15
        scale_y = 0.12

        # Sliding window
        while scale_x <= 1.75 and scale_y <= 1.75:
            image_resize = cv.resize(loaded_image, (0, 0), fx=scale_x, fy=scale_y)
            image_resize_gray = cv.cvtColor(image_resize, cv.COLOR_BGR2GRAY)
            image_resize_hog = hog(image_resize_gray, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2),
                                   feature_vector=False, orientations = 18)

            number_of_cols = image_resize.shape[1] // hog_pixels_per_cell[0] - 1
            number_of_rows = image_resize.shape[0] // hog_pixels_per_cell[0] - 1
            number_of_cell_in_template = train_window_size[0] // hog_pixels_per_cell[0] - 1

            # Slide across hog cells
            for y in range(0, number_of_rows - number_of_cell_in_template, sliding_window_step_size):
                for x in range(0, number_of_cols - number_of_cell_in_template, sliding_window_step_size):
                    x_min = int(x * hog_pixels_per_cell[1] * 1 // scale_x)
                    y_min = int(y * hog_pixels_per_cell[0] // scale_y)
                    x_max = int((x * hog_pixels_per_cell[1] + train_window_size[1]) * 1 // scale_x)
                    y_max = int((y * hog_pixels_per_cell[0] + train_window_size[0]) * 1 // scale_y)

                    # Check if image contains some skin color
                    if loaded_image_hsv_skin[y_min:y_max, x_min:x_max].mean() >= 90:
                        score = np.dot(image_resize_hog[y:y + number_of_cell_in_template,
                                       x:x + number_of_cell_in_template].flatten(), classifier.coef_.T) + \
                                classifier.intercept_[0]

                        # Append score
                        if score[0] > 0:
                            scores.append(score[0])
                            detections.append((x_min, y_min, x_max, y_max))

            scale_x *= 1.02
            scale_y *= 1.02

        # If there is a detection save it, if there are multiple detections run non_maximal_suppression to get the greatest square
        if len(detections) > 0:
            image_detections, image_scores = non_maximal_suppression(np.array(detections), np.array(scores),
                                                                     loaded_image.shape)

            # Save final detections and file paths
            for detection in image_detections:
                final_detections.append(detection)
                final_file_paths.append(file)
                cv.rectangle(loaded_image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 1)

            # Save scores
            for score in image_scores:
                final_scores.append(score)

        end_time = timeit.default_timer()


        if len(detections) > 0:
            print(
                f'Time to process test image {file_no + 1:3}/{length_of_files},    with detecion, is {end_time - start_time:f} sec.')
        else:
            print(
                f'Time to process test image {file_no + 1:3}/{length_of_files}, without detection, is {end_time - start_time:f} sec.')

    # Convert to numpy array final lists
    final_detections = np.asarray(final_detections)
    final_file_paths = np.asarray(final_file_paths)
    final_scores = np.asarray(final_scores)

    print(f"Total detections: {len(final_detections)}")

    if SAVE_FILES:
        np.save("./TEMP_Gheorghe_Radu-Mihai_351/task1/detections_all_faces.npy", final_detections)
        np.save("./TEMP_Gheorghe_Radu-Mihai_351/task1/file_names_all_faces.npy", final_file_paths)
        np.save("./TEMP_Gheorghe_Radu-Mihai_351/task1/scores_all_faces.npy", final_scores)

    # Evaluate detection
    show_detections_with_ground_truth(final_detections, final_scores, final_file_paths)

if __name__ == "__main__":
    main()