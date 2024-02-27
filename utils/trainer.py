"""
Module for training and evaluating a model on the BDD100K dataset.
"""
import os
import torch

from torch.utils.data import DataLoader

from .train_utils import Trainer
from .dataset import BddDataset
from .model import get_model
from .transformations import Compose, ToTensor, RandomHorizontalFlip
from .evaluation import evaluate_model


# Global Constants
# pylint: disable=no-member
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
BDD_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'bus', 'traffic light', 'traffic sign',
    'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider'
]


def train(config, model_save_path='./models'):

    """
    Train and evaluate a model.

    Parameters:
    - config (dict): Dictionary containing configuration parameters for training.
    - model_save_path (str): Path to save trained model weights.

    Returns:
    - None
    """
    lr = config['lr']
    batch_size = config['batch_size']
    epochs = config['epochs']
    base_dir = config['img_dir']
    labels_dir = config['label_dir']
    model_name = config['model']
    train_model_flag =  config['train']
    iou_threshold =  config['iou_threshold']
    prob_threshold =  config['prob_threshold']
    train_images =  config['train_images']
    val_images =  config['val_images']

    train_transform = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    num_classes = len(BDD_INSTANCE_CATEGORY_NAMES)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    # step 1: data

    train_set = BddDataset(data_dir=base_dir,
    			labels_dir=labels_dir,
    			transforms=train_transform,
    			train_images= train_images,
    			flag='train',
    			label_list=BDD_INSTANCE_CATEGORY_NAMES)
    val_set = BddDataset(data_dir=base_dir,labels_dir=labels_dir,
    			transforms=train_transform,
    			train_images=train_images,
    			flag='val',
    			label_list=BDD_INSTANCE_CATEGORY_NAMES)

    train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          collate_fn=lambda x: tuple(zip(*x)),
                          shuffle=True)
    val_loader = DataLoader(val_set,
                          batch_size=batch_size,
                          collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(model_name, num_classes)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = Trainer(model, optimizer, lr_scheduler, train_loader, val_loader, DEVICE)

    json_dir= os.path.join(labels_dir, "labels", 'bdd100k_labels_images_val.json')
    vis_dir= os.path.join(base_dir, "images", '100k', 'val')

    if train_model_flag:
    	# train and validation
        for epoch in range(epochs):
            t_loss = trainer.train()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {t_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch+1}.pth'))
        evaluate_model(model, DEVICE, vis_dir, json_dir,
               iou_threshold, prob_threshold, val_images,
               classes=BDD_INSTANCE_CATEGORY_NAMES)

        print("Training and evaluation completed.")

    else:
        evaluate_model(model, DEVICE, vis_dir, json_dir,
               iou_threshold, prob_threshold, val_images,
               classes=BDD_INSTANCE_CATEGORY_NAMES)

        print("Only inference done with pretrained weights.")
