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
    optional.add_argument('-e', '--epochs', dest='num_of_epochs', help='Amount of epochs for training.', default=100, type=int)
    optional.add_argument('-o', '--output', dest='out_folder', help='Folder for result images.', default='results')
    optional.add_argument('--no-gif', dest='create_gifs', help='If set, no gifs will be created.', action='store_false')
    optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.set_defaults(create_gifs=True)
    return parser.parse_args()

kernels = np.array([
    [
        [1, 0, -1], 
        [2, 0, -2], 
        [1, 0, -1]
    ],
    [
        [-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]
    ],
    [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ],
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
])

kernel_names = [
    'sobel1',
    'sobel2',
    'sobel3',
    'sobel4'
]

def main():
    # Initialization - argument parsing and stage preparations
    args = parse_args()
    out_folder = args.out_folder if args.out_folder.endswith('/') else args.out_folder + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Training
    _, gray_img, _, samples, labels = prepare_for_training(args.training_image, kernels)
    weights = train(samples, labels, num_of_epochs=args.num_of_epochs)
    
    # Animated matrix
    if args.create_gifs:
        for i, name in zip(range(weights.shape[1]), kernel_names):
            create_matrix_gif(weights[:, i], save_folder=out_folder+"gif_images/", out_image=out_folder+f"{name}/animated_training_matrix.gif")
    else:
        print("Skipping gif creation.")
    
    # Training results
    results = np.round(weights[-1,:], 2)
    print("Results: ")
    for result, kernel, name in zip(results, kernels, kernel_names):
        print("Original filter:", kernel, "Trained filter:", result, sep="\n")
        compare_images(convolution(gray_img, result), convolution(gray_img, kernel), 
                    out_image=out_folder+f"{name}/training_image_compare.jpg")
        print("\n")
    
    # Results on validation image
    _, _, validation_image = load_normalize_and_grayscale(args.validation_image)
    for result, kernel, name in zip(results, kernels, kernel_names):
        avg_diff = compare_images(convolution(validation_image, result), convolution(validation_image, kernel), 
                                out_image=out_folder+f"{name}/validation_image_compare.jpg")
        print(f"Average difference between validation image filtered with trained and original {name} kernel is {avg_diff}%")

if __name__ == "__main__":
    main()
