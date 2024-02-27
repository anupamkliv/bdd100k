"""
Module containing a custom PyTorch dataset for BDD100K dataset.
"""
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class BddDataset(Dataset):

    """
    Custom PyTorch dataset for BDD100K dataset.

    Parameters:
    - data_dir (str): Directory containing image data.
    - labels_dir (str): Directory containing label data.
    - transforms (callable): Transforms to be applied to the images and targets.
    - train_images (int): Number of images to be sampled for training.
    - flag (str): Flag indicating whether the dataset is for training or validation.
    - label_list (list): List of class labels.

    Returns:
    - img (PIL.Image): Image data.
    - target (dict): Dictionary containing bounding box coordinates and labels.
    """

    def __init__(self, data_dir,labels_dir, transforms, train_images, flag, label_list):

        self.data_dir = data_dir
        self.transforms = transforms
        self.train_images=train_images
        if flag == 'train':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'train')
            self.json_dir = os.path.join(labels_dir, "labels", 'bdd100k_labels_images_train.json')
        if flag == 'val':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'val')
            self.json_dir = os.path.join(labels_dir, "labels", 'bdd100k_labels_images_val.json')

        self.names = [name[:-4] for name in
            list(filter(lambda x: x.endswith(".jpg"),
                os.listdir(self.img_dir)))]

        self.label_data = json.load(open(self.json_dir, 'r', encoding='UTF-8'))
        self.label_list = label_list
        #Sample only 50 images from train
        if flag == 'train':
            self.names = self.names[:self.train_images]


    def __getitem__(self, index):

        """
        Get item method to retrieve image and target.

        Parameters:
        - index (int): Index of the image.

        Returns:
        - img (PIL.Image): Image data.
        - target (dict): Dictionary containing bounding box coordinates and labels.
        """

        name = self.names[index]
        path_img = os.path.join(self.img_dir, name + ".jpg")
        # load img
        img = Image.open(path_img).convert("RGB")
        # load boxes and label
        label_data = self.label_data
        points = label_data[index]['labels']
        boxes_list = []
        labels_list = []
        for point in points:
            if 'box2d' in point.keys():
                box = point['box2d']
                boxes_list.append([box['x1'], box['y1'], box['x2'], box['y2']])
                label = point['category']
                labels_list.append(self.label_list.index(label))
        # pylint: disable=no-member
        boxes = torch.tensor(boxes_list, dtype=torch.float)
        labels = torch.tensor(labels_list, dtype=torch.long)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        """
        Get length method to retrieve the total number of images in the dataset.

        Returns:
        - length (int): Total number of images in the dataset.
        """
        if len(self.names) == 0:
            raise Exception(f"Data directory {self.data_dir} is empty!")
        return len(self.names)
