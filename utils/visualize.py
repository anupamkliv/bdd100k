"""
Module for visualization functions related to object detection.

This module provides functions to visualize bounding boxes and class labels on images
generated by object detection models. It includes a function to visualize bounding
boxes with class labels and confidence scores.

Functions:
- vis_bbox: Visualizes bounding boxes and class labels on an image.
"""

import matplotlib.pyplot as plt


def vis_bbox(img, output_dict, classes, max_vis=40, prob_thres=0.02):
    """
    Visualizes bounding boxes and class labels on an image.

    Parameters:
    - img (numpy.ndarray or PIL.Image): The input image.
    - output_dict (dict): Dictionary containing the output from the object detection model.
                          It should contain keys "boxes", "scores", and "labels".
    - classes (list): List of class names.
    - max_vis (int): Maximum number of bounding boxes to visualize.
    - prob_thres (float): Probability threshold for filtering bounding boxes to visualize.

    Returns:
    - None
    """
    _, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')
    out_boxes = output_dict["boxes"].cpu()
    out_scores = output_dict["scores"].cpu()
    out_labels = output_dict["labels"].cpu()
    for idx in range(min(len(out_boxes), max_vis)):
        score = out_scores[idx].item()
        if score < prob_thres:
            continue
        bbox = out_boxes[idx].numpy()
        class_name = classes[out_labels[idx]]
        ax.add_patch(
        		plt.Rectangle((bbox[0], bbox[1]),
        		bbox[2] - bbox[0], bbox[3] - bbox[1],
        		fill=False, edgecolor='red',
        		linewidth=3.5))
        ax.text(
        		bbox[0], bbox[1] - 2,
        		f'{class_name} {score:.3f}',
        		bbox={'facecolor': 'blue',
        		'alpha': 0.5}, fontsize=14, color='white')
    plt.show()
    plt.close()