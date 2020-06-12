from images import prepare_for_training, convolution, compare_images, load_normalize_and_grayscale
from training import train
from matrix_gif import create_matrix_gif
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Supervised training of edge detection filters.', add_help=False)
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    
    required.add_argument('-t', '--train', dest='training_image', required=True, help='Image for training purposes.')
    required.add_argument('-v', '--validate', dest='validation_image', required=True, help='Image for valdiation purposes.')
    optional.add_argument('-o', '--output', dest='out_folder', help='Folder for result images.', default='results')
    optional.add_argument('-k', '--kernel', dest='kernel', help='Kernel for supervised training')
    optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    return parser.parse_args()

kernel = np.array([
    [1, 0, -1], 
    [2, 0, -2], 
    [1, 0, -1]
])

def main():
    # Initialization - argument parsing and stage preparations
    args = parse_args()
    out_folder = args.out_folder if args.out_folder.endswith('/') else args.out_folder + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Training
    _, gray_img, _, samples, labels = prepare_for_training(args.training_image, kernel)
    weights = train(samples, labels)
    
    # Animated matrix
    create_matrix_gif(weights, save_folder=out_folder+"gif_images/", out_image=out_folder+"animated_training_matrix.gif")
    
    # Training results
    result = np.round(weights[-1], 2)
    print("Original filter:", kernel, "Trained filter:", result, sep="\n")
    compare_images(convolution(gray_img, result), convolution(gray_img, kernel), 
                   out_image=out_folder+"training_image_compare.jpg")
    
    # Results on validation image
    _, _, validation_image = load_normalize_and_grayscale(args.validation_image)
    avg_diff = compare_images(convolution(validation_image, result), convolution(validation_image, kernel), 
                              out_image=out_folder+"validation_image_compare.jpg")
    print(f"Average difference between validation image filtered with trained and original kernel is {avg_diff}%")

if __name__ == "__main__":
    main()
