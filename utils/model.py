"""
Module to get the training model.
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(model_name, num_classes):
    """
    Returns an object detection model based on the model_name.

    Parameters:
    - model_name (str): The name of the model architecture.
    - num_classes (int): The number of classes, including the background class.

    Returns:
    - A PyTorch model ready for training.
    """
    if model_name == "fasterrcnn_resnet50_fpn":
        # Load a pre-trained Faster R-CNN model for fine-tuning
        #backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        #backbone.out_channels = 1280
        
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        			weights="DEFAULT",
        			pretrained=True, 
        			max_detections_per_img=50)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model
