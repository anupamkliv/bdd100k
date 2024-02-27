"""
Command-line utility to generate statistics on BDD100K dataset.

This script allows generating various statistics and visualizations on the BDD100K dataset.
"""
import argparse

from dataset_info.dataset_info import main as dataset_info_main

def main(arguments):
    """
    Train the model based on the provided arguments.

    Args:
        arguments (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    config = {
        'all_stats': arguments.all_stats,
        'train_stats_attribute': arguments.train_attribute,
        'valid_stats_attribute': arguments.valid_attribute,
        'imblance': arguments.imbalance,
        'unclear_image': arguments.unclear_image,
        'traffic_light': arguments.traffic_light,
        'area_stats': arguments.area_stats,
        'train_json': arguments.train_json,
        'label_json': arguments.label_json,
        'train_images': arguments.train_images,
        'label_images': arguments.label_images,
        'visualize_dataset': arguments.visualize_dataset,
        'number_visualize': arguments.number_visualize,
            }
    dataset_info_main(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistics on BDD100K dataset')
    parser.add_argument('--all_stats',
                        type=int, default=1,
                        help=' 1 if all stats are needed')
    parser.add_argument('--train_attribute',
                        type=int, default=0,
                        help='give no of images for train w.r.t attribute')
    parser.add_argument('--valid_attribute',
                         type=int, default=0,
                         help='give no of images for validation w.r.t attribute')
    parser.add_argument('--imbalance',
                        type=int, default=0,
                        help='give class and sample imbalance')
    parser.add_argument('--unclear_image',
                        type=int, default=0,
                        help='provide blur images present')
    parser.add_argument('--traffic_light',
                         type=int, default=0,
                         help='provide images where trafficlightcolor is None')
    parser.add_argument('--area_stats',
                        type=int, default=0,
                        help='provide variation in areas of BB')
    parser.add_argument('--visualize_dataset',
                        type=int,
                        default=1,
                        help='base_val_image directory')
    parser.add_argument('--number_visualize',
                        type=int, default=3,
                        help='# of images to be visulize. if 3 then 3x3')
    parser.add_argument('--train_json',
                        type=str,
                        default='dataset/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json',
                        help='base_image directory')
    parser.add_argument('--label_json',
                        type=str,
                        default='dataset/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json',
                        help='base_val_image directory')
    parser.add_argument('--train_images', 
                        type=str,
                        default='dataset/bdd100k_images_100k/bdd100k/images/100k/train',
                        help='base_image directory')
    parser.add_argument('--label_images',
                        type=str,
                        default='dataset/bdd100k_images_100k/bdd100k/images/100k/val',
                        help='base_val_image directory')
    args = parser.parse_args()
    main(args)
