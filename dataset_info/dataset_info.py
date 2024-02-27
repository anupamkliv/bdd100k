"""
This module contains functions for calculating dataset statistics.
"""

import os
import json
import random
import shutil

from .plot import *
from .utils import  *

def load_data(file_path):
    """
    Load data from a JSON file.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: The data loaded from the JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def count_attributes(data):
    """
    Count the number of images for each frame attribute in the dataset.

    Parameters:
    - data (list): A list of dictionaries representing data for each image, where each dictionary
      contains information about the attributes of the image.

    Returns:
    - tuple: A tuple containing dictionaries for weather, scene, and time of day attributes,
      where each dictionary maps attribute values to the number of images having that attribute.
    """
    weather, scene, timeofday = {}, {}, {}

    for item in data:
        w = item['attributes']['weather']
        s = item['attributes']['scene']
        t = item['attributes']['timeofday']
        weather[w] = weather.get(w, 0) + 1
        scene[s] = scene.get(s, 0) + 1
        timeofday[t] = timeofday.get(t, 0) + 1

    return weather, scene, timeofday


def print_counts_and_total(counts_dict, category_name):
    """
    Print counts and total for a given category, all in one line.

    Parameters:
    - counts_dict (dict): A dictionary containing counts of items for different categories.
    - category_name (str): The name of the category.

    Returns:
    - None
    """
    counts_str = ', '.join([f"{k}: {v}" for k, v in counts_dict.items()])
    print(
        f"{category_name.capitalize()}: {counts_str}. "
        f"Total images in {category_name}: {sum(counts_dict.values())}"
    )


def find_images_without_labels(images_folder, data):
    """
    Find images without labels and print their names and count.

    Parameters:
    - images_folder (str): The path to the folder containing images.
    - data (list): A list of dictionaries containing data about the images.

    Returns:
    - list: A list of filenames of images without labels.
    """
    json_image_names = {item["name"] for item in data}
    images_without_labels = [
        filename
        for filename in os.listdir(images_folder)
        if filename not in json_image_names
    ]
    #print("Images without labels in train.json:")
    for image_name in images_without_labels:
        print(image_name)
    print(f"Total images without labels: {len(images_without_labels)}")
    return images_without_labels


def move_images_to_folder(images, source_folder, destination_folder):
    """
    Move specified images to another folder.

    Parameters:
    - images (list): A list of filenames of images to be moved.
    - source_folder (str): The path to the source folder containing the images.
    - destination_folder (str): The path to the destination folder.

    Returns:
    - None
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for filename in images:
        shutil.move(
            os.path.join(source_folder, filename),
            os.path.join(destination_folder, filename)
        )
    print(f"Moved {len(images)} images to {destination_folder}")


def main(config):
    """
    Main function for generating dataset statistics.

    Parameters:
    - config (dict): A dictionary containing configuration parameters.

    Returns:
    - None
    """
    all_stats = config['all_stats']
    print("Dataset loading")
    train_json = config['train_json']
    train_json = config['train_json']
    train_data  = load_data(train_json)
    valid_json = config['label_json']
    valid_data = load_data(valid_json)
    images_folder_train  = config['train_images']
    images_folder_val = config['label_images']
    number_visualize = config['number_visualize']
    classes = [
    'bus',
    'traffic light',
    'traffic sign',
    'person',
    'bike',
    'truck',
    'motor',
    'car',
    'train',
    'rider'
    ]

    train_stats_attribute =  config['train_stats_attribute']
    valid_stats_attribute = config['valid_stats_attribute']
    imblance = config['imblance']
    unclear_image = config['unclear_image']
    traffic_light = config['traffic_light']
    area_stats = config['area_stats']
    visualize_dataset = config['visualize_dataset']
    save_images ='dataset_statistics'

    if not os.path.exists(save_images):
        os.makedirs(save_images)

    if train_stats_attribute == 1 or all_stats ==1:
        print("\nCounting the number of images present in the Training set for each attribute")
        (weather_counts_train,
        scene_counts_train,
        timeofday_counts_train) = count_attributes(train_data)
        print_counts_and_total(weather_counts_train, "weather")
        print_counts_and_total(scene_counts_train, "scene")
        print_counts_and_total(timeofday_counts_train, "timeofday")
        print("\nCounting images in train without labels")
        images_without_labels = find_images_without_labels(images_folder_train, train_data)
        move_images_to_folder(images_without_labels, images_folder_train, "images_without_labels")

    if valid_stats_attribute == 1 or all_stats ==1:
        print("\nCounting the number of images present in the Validation set for each attribute")
        (weather_counts_valid,
        scene_counts_valid,
        timeofday_counts_valid) = count_attributes(valid_data)

        print_counts_and_total(weather_counts_valid, "weather")
        print_counts_and_total(scene_counts_valid, "scene")
        print_counts_and_total(timeofday_counts_valid, "timeofday")

    if visualize_dataset ==1 or all_stats ==1:
        directory ='visualize'
        save_path = os.path.join(save_images, directory)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        visualize_validation_dataset_grid(
        images_folder_train,
        train_data,
        save_path,
        number_visualize
        )

        visualize_validation_dataset_grid(
        images_folder_val,
        valid_data,
        save_path,
        number_visualize)
        print("\n\n Visualize dataset done")

    if imblance == 1 or all_stats ==1:
        directory ='imbalance'
        save_path = os.path.join(save_images, directory)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plot_histogram(train_data, 'Histogram of freq of unique class in train set' , save_path)
        plot_histogram(valid_data, 'Histogram of freq of unique class in validation set', save_path)
        print("Class statistics plot done")

        plot_histogram_images_per_class(
            train_data,
            "Number of images per class for training set",
            save_path
        )

        plot_histogram_images_per_class(
            valid_data,
            "Number of images per class for validation set",
            save_path
        )

        print("\nClass imbalance plot done")

    if unclear_image == 1 or all_stats ==1:
        directory ='blur'
        save_path = os.path.join(save_images, directory)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print("\nCalculating unclear images in train and validaton set")
        unclear_images_train = find_unclear_images(images_folder_train, clarity_threshold=10)
        unclear_images_valid = find_unclear_images(images_folder_val, clarity_threshold=10)

        (weather_counts_train,
        scene_counts_train,
        timeofday_counts_train) = count_attribute_occurrences(unclear_images_train, train_data)
        (weather_counts_valid,
        scene_counts_valid,
        timeofday_counts_valid) = count_attribute_occurrences(unclear_images_valid, valid_data)

        print(f"Total unclear images found in train: {len(unclear_images_train)}/69863")
        print("Unclear Images by Weather:")
        print(" ".join([
            f"{weather.capitalize()}: {count}" 
            for weather, count in weather_counts_train.items()]))
        print("\nUnclear Images by Scene:")
        print(" ".join([
            f"{scene.capitalize()}: {count}" 
            for scene, count in scene_counts_train.items()]))
        print("\nUnclear Images by Time of Day:")
        print(" ".join([
            f"{timeofday.capitalize()}: {count}" 
            for timeofday, count in timeofday_counts_train.items()]))

        print(f"\nTotal unclear images found in valid: {len(unclear_images_valid)}/10000")
        print("Unclear Images by Weather:")
        print(" ".join([
            f"{weather.capitalize()}: {count}" 
            for weather, count in weather_counts_valid.items()]))
        print("\nUnclear Images by Scene:")
        print(" ".join([
            f"{scene.capitalize()}: {count}" 
            for scene, count in scene_counts_valid.items()]))
        print("\nUnclear Images by Time of Day:")
        print(" ".join([
            f"{timeofday.capitalize()}: {count}"
            for timeofday, count in timeofday_counts_valid.items()]))

        print("Statistics of unclear images plotted")

        plot_attribute_histogram(
            weather_counts_train,
            weather_counts_valid,
            'weather', save_path)
        plot_attribute_histogram(
            scene_counts_train,
            scene_counts_valid,
            'scene', save_path)
        plot_attribute_histogram(
            timeofday_counts_train,
            timeofday_counts_valid,
            'time of day', save_path)
        plot_attribute_histogram(
            timeofday_counts_train,
            timeofday_counts_valid,
            'time of day',
            save_path)

    if traffic_light == 1 or all_stats ==1:
        directory ='traffic_light'
        save_path = os.path.join(save_images, directory)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print("\n\nNumber of images for where the traffic light has no color")
        (none_traffic_sign_count_train,
        none_traffic_sign_count_valid,
        none_traffic_sign_images_train,
        none_traffic_sign_images_val) = count_none_traffic_light_color(train_data, valid_data)

        print(f"Taining images with no trafficLightColor: {none_traffic_sign_count_train}")
        print(f"Validation images with no trafficLightColor: {none_traffic_sign_count_valid}")

        random_images_train = random.sample(none_traffic_sign_images_train, 3)
        random_images_val = random.sample(none_traffic_sign_images_val, 3)

        display_with_boxes(random_images_train,
        'Randomly Selected Images from Train Set',
        images_folder_train, train_data, save_path)
        display_with_boxes(random_images_val,
        'Randomly Selected Images from Validation Set',
        images_folder_val, valid_data, save_path)

        print("3 random images with BB saved for both train and valid")

    if area_stats == 1 or all_stats ==1:
        directory ='area_stats'
        save_path = os.path.join(save_images, directory)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_stats = calculate_area_stats(train_json, classes)
        valid_stats = calculate_area_stats(valid_json, classes)
        plot_mean_std(train_stats, valid_stats, classes, save_path, 'mean_std_plot.png')
        print("\n\n Area plot done")
