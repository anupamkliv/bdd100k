"""
Module for plotting various visualizations related to object detection evaluation.

Functions:
- plot_histogram(): histogram showing class imbalance.
- plot_histogram_images_per_class(): histogram for the number of images per class.
- plot_attribute_histogram(): histogram for the number of unclear images.
- display_with_boxes(): Display images with bounding boxes where the traffic light color is None.
- visualize_validation_dataset_grid(): Visualize images with bounding boxes.
- plot_mean_std(): Bar chart showing the mean and standard deviation of BB areas for each class.
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib import patches
from PIL import Image


def plot_histogram(data, title, save_images):
    """
    Plot and save a histogram showing the class imbalance.

    Parameters:
    - data (list): List of dictionaries containing image labels.
    - title (str): Title for the histogram plot.
    - save_images (str): Directory to save the generated histogram image.

    Returns:
    - None
    """
    num_classes_per_image = [
        len(set(label['category']
        for label in item.get('labels', []))) for item in data]

    plt.figure()
    plt.hist(num_classes_per_image,
            bins=range(1, max(num_classes_per_image) + 2),
            edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Classes')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(range(1, max(num_classes_per_image) + 1))
    filename = title+".png"
    filename = os.path.join(save_images, filename)
    plt.savefig(filename)
    plt.close()


def plot_histogram_images_per_class(train_data, title, save_images):
    """
    Plot and save a histogram for the number of images per class.

    Parameters:
    - train_data (list): List of dictionaries containing image labels for the training dataset.
    - title (str): Title for the histogram plot.
    - save_images (str): Directory to save the generated histogram image.

    Returns:
    - None
    """
    class_counts = {}
    for item in train_data:
        for label in item.get('labels', []):
            category = label.get('category')
            class_counts[category] = class_counts.get(category, 0) + 1
    plt.figure()
    plt.bar(class_counts.keys(), class_counts.values(), edgecolor='black', alpha=0.7)
    plt.xlabel('Class Name')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(rotation=90)
    filename = title+".png"
    # Annotate each bar with its count
    for i, count in enumerate(class_counts.values()):
        plt.text(i, count, str(count), ha='center', va='bottom', rotation=0, fontsize=8)
    plt.tight_layout(pad=3.0)
    filename = os.path.join(save_images, filename)
    plt.savefig(filename)
    plt.close()


def plot_attribute_histogram(train_counts, valid_counts, attribute_name, save_path):
    """
    Plot and save a histogram showing the number of unclear images w.r.t. the specified attribute.

    Parameters:
    - train_counts (dict): Dic with counts of blur images for each attribute in the training set.
    - valid_counts (dict): Dic with counts of blur images for each attribute in the validation set.
    - attribute_name (str): Name of the attribute for which the histogram is plotted.
    - save_path (str): Directory to save the generated histogram image.

    Returns:
    - None
    """
    attributes = list(train_counts.keys())
    train_values, valid_values = list(train_counts.values()), list(valid_counts.values())

    x = range(len(attributes))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, train_values, width, label='Train')
    rects2 = ax.bar([i + width for i in x], valid_values, width, label='Valid')

    ax.set_ylabel('Count')
    ax.set_title(f'Unclear Images by {attribute_name.capitalize()}')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(attributes, rotation=45, ha='right')
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    filename = attribute_name+".png"
    filename = os.path.join(save_path, filename)
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


def display_with_boxes(image_list, title, folder_name, data, save_path):
    """
    Display images with bounding boxes for traffic lights that are not labeled as 'none'.

    Parameters:
    - image_list (list): List of image names to display.
    - title (str): Title for the display.
    - folder_name (str): Path to the folder containing the images.
    - data (list): List of dictionaries containing labels for the images.
    - save_path (str): Directory to save the displayed images.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 4))
    flag = folder_name.split('/')[-1]
    for i, image_name in enumerate(image_list, 1):
        plt.subplot(1, 3, i)
        base_name = os.path.splitext(image_name)[0]
        new_image_name = f"{flag}_{base_name}"
        img = mpimg.imread(folder_name+"/"+image_name)  # Load the image
        plt.imshow(img)
        plt.title(image_name)
        plt.axis('on')
        found_label = False  # Flag to track if label is found for the current image
        # Load labels for the current image
        for item in data:
            if item['name'] == image_name:
                labels = item.get('labels', [])
                for label in labels:
                    if label['category'] == 'traffic light' and \
                       label['attributes'].get('trafficLightColor') != 'none':

                        # Extract bounding box coordinates
                        x1, y1 = label['box2d']['x1'], label['box2d']['y1']
                        x2, y2 = label['box2d']['x2'], label['box2d']['y2']
                        # Create a rectangle patch with red color and thick lines
                        rect = patches.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            linewidth=3,
                            edgecolor='r',
                            facecolor='none'
                           )
                        # Add the rectangle patch to the current axis
                        plt.gca().add_patch(rect)
                        found_label = True  # Set flag to True since label is found
                        break  # Break to count only once per image
                break  # Break once labels for the current image are found
        if not found_label:
            print(f"No label found for {image_name} in {flag}")
    plt.suptitle(title, fontsize=16)
    path = new_image_name+".jpg"
    filename = os.path.join(save_path, path)
    plt.savefig(filename)


def visualize_validation_dataset_grid(image_dir, labels_data, save_path, num_rows=1):
    """
    Visualize images with bounding boxes.

    Parameters:
    - image_dir (str): Directory path containing the images.
    - labels_data (list): List of dictionaries containing labels for the images.
    - save_path (str): Directory to save the visualization images.
    - num_rows (int): Number of rows in the visualization grid. Default is 1.

    Returns:
    - None
    """
    num_columns = num_rows
    num_images = num_rows * num_columns
    image_names = random.sample(os.listdir(image_dir), num_images)
    last_word = image_dir.split('/')[-1]+"_"

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 5 * num_rows))
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()

    for i, image_name in enumerate(image_names):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        axs[i].imshow(image)

        # Matching labels for the current image
        image_labels = [label for label in labels_data if label["name"] == image_name]

        for label in image_labels:
            for obj in label["labels"]:
                category = obj["category"]
                if category == "drivable area":
                    continue  # Ignore drivable area
                box = obj.get("box2d", None)
                if box:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    rect = patches.Rectangle(
                            (x1, y1),
                             x2 - x1,
                             y2 - y1,
                             linewidth=2,
                             edgecolor='r',
                             facecolor='none')
                    axs[i].add_patch(rect)
                    axs[i].text(x1, y1 - 10, category, fontsize=12, color='r')

        axs[i].axis('off')
        axs[i].set_title(image_name)

    image_name =last_word+"_"+image_name
    filename = os.path.join(save_path, image_name)
    plt.savefig(filename)


def plot_mean_std(train_stats, valid_stats, classes, save_path,  image_name='mean_std_plot.png'):
    """
    Plot the mean and standard deviation of the areas of bounding boxes for each class.

    Parameters:
    - train_stats (dict): Dict with mean and std of BB areas for each class in the training set.
    - valid_stats (dict): Dict with mean and std of BB areas for each class in the validation set.
    - classes (list): List of class names.
    - save_path (str): Directory path to save the plot image.
    - image_name (str): Name of the image file to be saved. Default is 'mean_std_plot.png'.

    Returns:
    - None
    """
    x = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    train_means = [train_stats[cls]['mean'] for cls in classes]
    train_stds = [train_stats[cls]['std'] for cls in classes]
    valid_means = [valid_stats[cls]['mean'] for cls in classes]
    valid_stds = [valid_stats[cls]['std'] for cls in classes]

    rects1 = ax.bar(x - width/2,
                    train_means, width,
                    label='Train Mean', yerr=train_stds,
                    capsize=5)
    rects2 = ax.bar(x + width/2,
                    valid_means,width,
                    label='Validation Mean', yerr=valid_stds,
                    capsize=5)

    ax.set_xlabel('Classes')
    ax.set_ylabel('Mean Area')
    ax.set_title('Mean and Standard Deviation of Area by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='upper right')
    filename = os.path.join(save_path, image_name)

    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
