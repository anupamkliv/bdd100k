"""
This module contains functions for analyzing and visualizing image data.

Functions:
- image_clarity_score(): Compute the clarity score of an image using edge detection.
- find_unclear_images(): Find images with clarity scores below a threshold.
- get_attributes(): Extract attribute parameters from image labels.
- count_attribute_occurrences(): Count occurrences of attributes in a list of images.
- count_none_traffic_light_color(): Count images with traffic lights having no color.
- calculate_area_stats(): Calculate statistics on areas covered by bounding boxes.
"""
import os
import json
import cv2
import numpy as np


def image_clarity_score(image_path):
    """
    Compute an image clarity score using edge detection.
    
    Parameters:
    - image_path (str): The path to the input image.
    Returns:
    - float: The computed image clarity score.

    The image clarity score is computed by performing edge detection on the input image using the Canny edge detection algorithm. 
    """
    image = cv2.imread(image_path, 0)
    edges = cv2.Canny(image, 100, 200)
    return np.mean(edges)

def find_unclear_images(directory, clarity_threshold=10):
    """
    Scan through images and list those with a clarity score below the threshold.

    Parameters:
    - directory (str): The directory containing the images to be scanned.
    - clarity_threshold (float, optional): The clarity threshold below which images are considered unclear.
      Defaults to 10.

    Returns:
    - list: A list of paths to the unclear images.
    """
    unclear_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                score = image_clarity_score(image_path)
                if score < clarity_threshold:
                    unclear_images.append(image_path)
    return unclear_images


def get_attributes(image_path, labels_data):
    """
    Get the parameters of the attributes present in the image from the labels data.

    Parameters:
    - image_path (str): The path to the image for which attributes are to be retrieved.
    - labels_data (list): A list containing label data for multiple images.

    Returns:
    - dict: A dictionary containing the parameters of the attributes present in the image.
    """
    image_name = os.path.basename(image_path)
    image_attributes = {}
    for item in labels_data:
        if item["name"] == image_name:
            image_attributes = item.get("attributes", {})
            break
    return image_attributes


def count_attribute_occurrences(images, data):
    """
    Count the number of images for each attribute.

    Parameters:
    - images (list): A list of image paths.
    - data (list): A list containing label data for multiple images.

    Returns:
    - tuple: A tuple containing three dictionaries, each representing the count of images for each attribute.
             The dictionaries are structured as follows:
             - The first dictionary contains counts for weather attributes.
             - The second dictionary contains counts for scene attributes.
             - The third dictionary contains counts for time of day attributes.
    """
    weather, scene, timeofday = {}, {}, {}

    for image_path in images:
        attributes = get_attributes(image_path, data)
        w = attributes.get("weather", "undefined")
        s = attributes.get("scene", "undefined")
        t = attributes.get("timeofday", "undefined")
        weather[w] = weather.get(w, 0) + 1
        scene[s] = scene.get(s, 0) + 1
        timeofday[t] = timeofday.get(t, 0) + 1

    return weather, scene, timeofday

def count_none_traffic_light_color(train_data, valid_data):
    """
    Count the images where the traffic light color attribute is 'none'.

    Parameters:
    - train_data (list): A list containing label data for training images.
    - valid_data (list): A list containing label data for validation images.

    Returns:
    - tuple: A tuple containing four elements:
             1. The count of images in the training set where the traffic light color attribute is 'none'.
             2. The count of images in the validation set where the traffic light color attribute is 'none'.
             3. A list of image names in the training set where the traffic light color attribute is 'none'.
             4. A list of image names in the validation set where the traffic light color attribute is 'none'.
    """
    none_traffic_sign_count_train, none_traffic_sign_count_valid = 0, 0
    none_traffic_sign_images_train,none_traffic_sign_images_val  = [],[]

    for data in (train_data, valid_data):
        for item in data:
            labels = item.get('labels', [])
            for label in labels:
                if label['category'] == 'traffic light' and \
                    label['attributes'].get('trafficLightColor') == 'none':

                    if data == train_data:
                        none_traffic_sign_count_train += 1
                        none_traffic_sign_images_train.append(item['name'])
                    else:
                        none_traffic_sign_count_valid += 1
                        none_traffic_sign_images_val.append(item['name'])
                    break

    return (
        none_traffic_sign_count_train,
        none_traffic_sign_count_valid,
        none_traffic_sign_images_train,
        none_traffic_sign_images_val
    )


def calculate_area_stats(json_file, classes):
    """
    Calculate statistics on the areas covered by bounding boxes for specified classes.

    Parameters:
    - json_file (str): The path to the JSON file containing label data.
    - classes (list): A list of class names for which to calculate statistics.

    Returns:
    - dict: A dictionary containing statistics for each class. Each key-value pair
            corresponds to a class and its associated statistics, including mean,
            standard deviation, minimum, and maximum area covered by bounding boxes.
    """
    with open(json_file) as f:
        data = json.load(f)

    stats = {cls: [] for cls in classes}

    for item in data:
        for label in item.get("labels", []):
            category = label.get("category")
            if category in classes:
                box = label.get("box2d")
                if box:
                    area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                    stats[category].append(area)

    for cls, areas in stats.items():
        area_array = np.array(areas)
        stats[cls] = {
            'mean': np.mean(area_array),
            'std': np.std(area_array),
            'min': np.min(area_array),
            'max': np.max(area_array)
        }

    return stats
