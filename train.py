"""
Train a model on BDD100K dataset.
"""
import argparse

from utils.trainer import train


def main(arguments):
    """
    Train the model based on the provided arguments.

    Args:
        arguments (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    config = {
        'lr': arguments.lr,
        'batch_size': arguments.batch_size,
        'epochs': arguments.epochs,
        'model': arguments.model,
        'img_dir': arguments.img_dir,
        'label_dir': arguments.label_dir,
        'train': arguments.train,
        'prob_threshold': arguments.prob_thres,
        'iou_threshold': arguments.iou_thres,
        'train_images': arguments.train_images,
        'val_images': arguments.val_images,
    }
    train(config)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on BDD100K dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=2, help='max epoch')
    parser.add_argument('--model', type=str, default='fasterrcnn_resnet50_fpn', help='model name')
    parser.add_argument('--img_dir', type=str, default='dataset/bdd100k_images_100k/bdd100k/', help='base_image directory')
    parser.add_argument('--label_dir', type=str, default='dataset/bdd100k_labels_release/bdd100k/', help='base_val_image directory')
    parser.add_argument('--train', type=int, default=1, help='flag to 0 if only inference')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='thresshold for AP')
    parser.add_argument('--prob_thres', type=float, default=0.02, help='thresshold to visualize BB')
    parser.add_argument('--train_images', type=int, default=50, help=' #images during training')
    parser.add_argument('--val_images', type=int, default=3, help='#images during inference')
    args = parser.parse_args()
    main(args)
