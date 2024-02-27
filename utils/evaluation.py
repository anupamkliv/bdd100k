"""
Module for evulation
"""
import os
import random
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import auc

from .visualize import vis_bbox


def count_unique_classes_gt(gt_boxes):

    """
    Count the number of occurrences of each class in the ground truth boxes.

    Parameters:
    - gt_boxes (list): List of tuples containing class name and bounding box coordinates.

    Returns:
    - class_counts (dict): Dictionary containing class names as keys and their counts as values.
    """
    class_counts = {}
    for category, _ in gt_boxes:
        class_counts[category] = class_counts.get(category, 0) + 1
    return class_counts


def count_unique_classes_pred(pred_class_index, classes):
    """
    Count the number of occurrences of each class in the predicted labels.

    Parameters:
    - pred_labels (tensor): Tensor containing predicted class indices.
    - classes (list): List of class names.

    Returns:
    - class_counts (dict): Dictionary containing class names as keys and their counts as values.
    """
    # Convert tensor to list
    class_list = pred_class_index.tolist()

    # Count occurrences of each unique class
    class_counts = {}
    for class_index in class_list:
        class_counts[class_index] = class_counts.get(class_index, 0) + 1

    # Map class indices to class names
    class_counts_with_names = {
        classes[class_index]: count
        for class_index, count in class_counts.items()
        }

    return class_counts_with_names


def limit_predictions(output, max_detections_per_img=50):
    """
    Limit the number of predictions stored in the model's output to max_detections_per_img.
    """
    num_images = len(output)
    for i in range(num_images):
        scores = output[i]['scores']
        num_predictions = min(len(scores), max_detections_per_img)
        top_scores, indices = scores.topk(num_predictions)
        output[i]['scores'] = top_scores
        output[i]['boxes'] = output[i]['boxes'][indices]
        output[i]['labels'] = output[i]['labels'][indices]
    return output



def evaluate_model(model, device, vis_dir, json_dir, iou_threshold, prob_threshold, num_images, classes):

    """
    Evaluate the model's performance on a dataset.

    Parameters:
    - model (torch.nn.Module): The trained model to be evaluated.
    - device (torch.device): The device on which the model will be evaluated.
    - vis_dir (str): Path to the directory containing visualization images.
    - json_dir (str): Path to the JSON file containing ground truth labels.
    - iou_threshold (float): IoU threshold for considering true positives.
    - prob_threshold (float): Probability threshold for considering predictions.
    - num_images (int): Number of images to evaluate.
    - classes (list): List of class names.

    Returns:
    - None
    """

    model.eval()
    img_names = [img_name for img_name in os.listdir(vis_dir) if img_name.endswith(".jpg")]
    random.shuffle(img_names)
    img_names = img_names[:num_images]

    preprocess = transforms.Compose([transforms.ToTensor()])

    total_iou = 0.0
    total_boxes = 0
    aps,final_iou= [],[]

    with open(json_dir, 'r', encoding='utf-8') as file_json:
        label_data = json.load(file_json)
    print(f"Images used from validation data: {num_images}/{len(label_data)}")

    for img_name in img_names:
        all_ious, all_labels, all_scores= [],[],[]
        path_img = os.path.join(vis_dir, img_name)
        input_image = Image.open(path_img).convert("RGB")
        img_tensor = preprocess(input_image).to(device)

        with torch.no_grad():
            output = model([img_tensor])
            output = limit_predictions(output, max_detections_per_img=50)
            print(f"\n \n Image: {img_name}")
            print(f"\n Model output-> No of boxes: {len(output[0]['boxes'])},"
                f" No of labels: {len(output[0]['labels'])},"
                f" No of scores: {len(output[0]['scores'])}")

        # Extract ground truth bounding boxes from JSON file
        gt_boxes = []
        for label_info in label_data:
            if label_info['name'] == img_name:
                #print("\n label_info['name']: ", label_info['name'])
                for label in label_info['labels']:

                    category = label['category']

                    if category in ('drivable area', 'lane'):
                        continue  # Skip processing drivable area labels
                    if 'box2d' in label:
                        box = label['box2d']
                        #print("Box found")
                        gt_boxes.append((category, (box['x1'], box['y1'], box['x2'], box['y2'])))


        #Count the number of boxes
        class_counts_predicted = count_unique_classes_pred(output[0]['labels'], classes)
        class_counts_original = count_unique_classes_gt(gt_boxes)

        print("Class names and counts")
        print("Predicted: ",class_counts_predicted)
        print("Original {",
                ",".join([f" '{class_name}': {count}" 
                for class_name, count in class_counts_original.items()]),"}")

        all_ious, all_labels, all_scores, computed_ious, total_iou, total_boxes = \
            compute_iou_metrics(input_image, output, classes, gt_boxes, prob_threshold, img_name)

        final_iou.append(computed_ious)
        total_boxes+=total_boxes
        total_iou+=total_iou

        average_precision = compute_ap_metrics(all_ious, all_labels, all_scores,
                                               classes, gt_boxes, iou_threshold)

        aps.append(average_precision)

    # Compute mean Average Precision (mAP)
    map_value = np.mean(aps)
    mean_iou =np.mean(final_iou)
    print(f"\n\nFinal mAP: {map_value:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")


    # Calculate average IoU if total_boxes is not zero
    if total_boxes != 0:
        average_iou = total_iou / total_boxes
        print(f"Average IoU: {average_iou:.4f}")
    else:
        print("\nNo predicted boxes found, cannot compute average IoU")


def compute_iou_metrics(input_image, output,classes, gt_boxes, prob_threshold,img_name):
    """
    Compute IoU metrics including IoU for each prediction, average IoU for the image,
    and visualization of bounding boxes with scores.

    Parameters:
    - input_image (PIL.Image): Input image.
    - output (list): List containing model output predictions.
    - classes (list): List of class names.
    - gt_boxes (list): List of ground truth bounding boxes.
    - prob_threshold (float): Probability threshold for considering predictions.

    Returns:
    - all_ious (list): List of computed IoU values for each prediction.
    - all_labels (list): List of predicted labels.
    - all_scores (list): List of prediction scores.
    - img_iou (float): Average IoU for the image.
    - total_iou (float): Total IoU across all predictions.
    - total_boxes (int): Total number of predicted boxes.
    """

    try:
        all_ious, all_labels, all_scores, img_ious= [],[],[],[]
        i=0
        total_iou =0
        total_boxes=0

    	# Compute IoU for each prediction and ground truth box pair
        pred_boxes = output[0]['boxes'].cpu().numpy()
        pred_scores = output[0]['scores'].cpu().numpy()

        for pred_box, pred_score in zip(pred_boxes, pred_scores):
            pred_class_index = output[0]['labels'][i]
            pred_class = classes[pred_class_index]
            max_iou = 0

            for gt_class, gt_box in gt_boxes:
                if pred_class == gt_class:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        total_boxes += 1
            all_ious.append(max_iou)
            all_labels.append(pred_class_index)
            all_scores.append(pred_score)

            total_iou += max_iou
            img_ious.append(max_iou)
            #print(f"pred class: {pred_class} gt_class: {gt_class}")
            i=i+1
    	# Print IoU for the current image
    	#print(f"Only IoU: {img_iou}, total_boxes: {total_boxes}")
        img_iou = sum(img_ious) / len(img_ious) if len(img_ious) > 0 else 0
        print(f"IoU for image: {img_iou:.4f}, total boxes: {total_boxes}")

    	# Visualize bounding boxes with scores
        vis_bbox(input_image, output[0], classes, img_name, max_vis=20, prob_thres=prob_threshold)
    except ZeroDivisionError:
        print(f"Skipping image {img_name} due to zero ground truth boxes")
    return all_ious, all_labels, all_scores, img_iou, total_iou, total_boxes


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (tuple): Tuple representing (x1, y1, x2, y2) of the first bounding box.
    - box2 (tuple): Tuple representing (x1, y1, x2, y2) of the second bounding box.

    Returns:
    - iou (float): Intersection over Union (IoU) value.
    """
    # Determine the coordinates of the intersection rectangle
    top_left_x = max(box1[0], box2[0])
    top_left_y = max(box1[1], box2[1])
    bottom_right_x = min(box1[2], box2[2])
    bottom_right_y = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, bottom_right_x - top_left_x + 1) * \
                        max(0, bottom_right_y - top_left_y + 1)


    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the Union area by subtracting the intersection area
    # from the sum of area of both bounding boxes
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


def compute_ap_metrics(all_ious, all_labels, all_scores, classes, gt_boxes, iou_threshold=0.5):
    """
    Computes and prints ap.

    Parameters:
    - all_ious (list): All computed IoUs.
    - all_labels (list): All predicted labels.
    - all_scores (list): All prediction scores.
    - classes (list): List of class names.
    - iou_threshold (float): IoU threshold for considering true positives.
    """
    # Calculate mAP
    # Compute precision-recall curves for each class
    # Sort predictions by confidence score (descending order)

    #print("Length of all labels: ", len(all_labels))
    sorted_indices = np.argsort(all_scores)[::-1]

    #print(f"sorted_indices: {sorted_indices}, length: {len(sorted_indices)}")
    sorted_ious = np.array(all_ious)[sorted_indices]

    #print(f"length: {len(sorted_ious)}")
    sorted_labels = np.array([label.cpu().item() for label in all_labels])[sorted_indices]

    #print(f"sorted_labels: {sorted_labels}, length: {len(sorted_labels)}")
    true_positive = np.zeros_like(sorted_ious)
    false_positive = np.zeros_like(sorted_ious)

    precision = np.zeros_like(sorted_ious, dtype=np.float64)
    recall = np.zeros_like(sorted_ious, dtype=np.float64)

    # Compute precision and recall at each prediction
    for i, _ in enumerate(sorted_ious):
        if classes[sorted_labels[i]] in classes:
            if sorted_ious[i] >= iou_threshold:
                true_positive[i] = 1
            else:
                false_positive[i] = 1
        precision[i] = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive)) \
                        if (np.sum(true_positive) + np.sum(false_positive)) > 0 else 0.0

        recall[i] = np.sum(true_positive) / len(gt_boxes)

    # Compute Average Precision (AP) for each class
    if len(precision) >= 2:  # Ensure there are at least two points to compute AUC
        auc_value = auc(recall, precision)
        print(f"AP for class: {auc_value:.4f}")
    else:
        auc_value = 0.0
        print("Insufficient data points to compute AP")

    return auc_value
