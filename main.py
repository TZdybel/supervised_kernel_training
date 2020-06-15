from images import prepare_for_training, convolution, compare_images, load_normalize_and_grayscale
from training import train
from matrix_gif import create_matrix_gif
import numpy as np
import argparse
import os
import sys
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Supervised training of edge detection filters.', add_help=False)
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    
    required.add_argument('-t', '--train', dest='training_image', required=True, help='Image for training purposes.')
    required.add_argument('-v', '--validate', dest='validation_image', required=True, help='Image for valdiation purposes.')
    optional.add_argument('-e', '--epochs', dest='num_of_epochs', help='Amount of epochs for training. By default set to 200.', 
                          default=200, type=int)
    optional.add_argument('-b', '--batch', dest='batch_size', help='Batch size for training. By default set to 1024.', 
                          default=1024, type=int)
    optional.add_argument('--acceptance', dest='acceptance_error', help='Value in percentage of acceptance error. By default set to 0.2.', 
                          default=0.2, type=float)
    optional.add_argument('-o', '--output', dest='out_folder', help='Folder for result images. By default set to "results"', 
                          default='results')
    optional.add_argument('--with-gif', dest='create_gifs', help='If set, no gifs will be created.', action='store_true')
    optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.set_defaults(create_gifs=False)
    return parser.parse_args()

kernels = np.array([[
        [1, 0, -1], 
        [2, 0, -2], 
        [1, 0, -1]
    ],[
        [-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]
    ],[
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ],[
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ],[
        [0, 1, 2],
        [-1, 0, 1],
        [-2, -1, 0]
    ],[
        [0, -1, -2],
        [1, 0, -1],
        [2, 1, 0]
    ],[
        [2, 1, 0],
        [1, 0, -1],
        [0, -1, -2]
    ],[
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2]
    ],[
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]
    ],[
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]
    ],[
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3]
    ],[
        [-3, -10, -3],
        [0, 0, 0],
        [3, 10, 3]
    ],[
        [0, 3, 10],
        [-3, 0, 3],
        [-10, -3, 0]
    ],[
        [0, -3, -10],
        [3, 0, -3],
        [10, 3, 0]
    ],[
        [10, 3, 0],
        [3, 0, -3],
        [0, -3, -10]
    ],[
        [-10, -3, 0],
        [-3, 0, 3],
        [0, 3, 10]
    ]
])

kernel_names = [
    'sobel_right',
    'sobel_left',
    'sobel_down',
    'sobel_up',
    'sobel_left_down',
    'sobel_right_up',
    'sobel_right_down',
    'sobel_left_up',
    'scharr_right',
    'scharr_left',
    'scharr_down',
    'scharr_up',
    'scharr_left_up',
    'scharr_right_up',
    'scharr_right_down',
    'scharr_left_down'
]

def main():
    # Initialization - argument parsing, logging and stage preparations
    logging.basicConfig(format='%(asctime)s %(message)s', filename='logs.log',level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    args = parse_args()
    out_folder = args.out_folder if args.out_folder.endswith('/') else args.out_folder + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Training
    logging.info("Preparing images for training...")
    _, gray_img, _, samples, labels = prepare_for_training(args.training_image, kernels)
    logging.info("Starting training process...")
    weights = train(samples, labels, num_of_epochs=args.num_of_epochs, batch_size=args.batch_size)
    logging.info("Training done.")
    
    # Training results
    results = np.round(weights[-1,:], 2)
    logging.info("Results: ")
    for result, kernel, name in zip(results, kernels, kernel_names):
        logging.info(f"Original {name} filter: \n{kernel}")
        logging.info(f"Trained {name} filter: \n{result}")
        
        out_image = out_folder+f"{name}/training_image_compare.jpg"
        logging.info(f"Saving training comparition to {out_image}\n")
        compare_images(convolution(gray_img, result), convolution(gray_img, kernel), 
                    out_image=out_image)
    
    # Results on validation image
    _, _, validation_image = load_normalize_and_grayscale(args.validation_image)
    logging.info(f"Acceptance error is set to {args.acceptance_error}% of distance between highest and lowest possible value")
    for result, kernel, name in zip(results, kernels, kernel_names):
        out_image = out_folder+f"{name}/validation_image_compare.jpg"
        logging.info(f"Saving validation comparition to {out_image}")
        perc_of_similarity = compare_images(convolution(validation_image, result), 
                                  convolution(validation_image, kernel), out_image=out_image, 
                                  acceptance_error=args.acceptance_error)
        logging.info(f"Similarity between validation image filtered with trained and original {name} kernel is {perc_of_similarity}%")
        
    # Animated matrix
    if args.create_gifs:
        for i, name in zip(range(weights.shape[1]), kernel_names):
            out_image = out_folder+f"{name}/animated_training_matrix.gif"
            logging.info(f"Creating matrix gif for {name}...")
            create_matrix_gif(weights[:, i], save_folder=out_folder+"gif_images/", 
                              out_image=out_folder+f"{name}/animated_training_matrix.gif")
    else:
        logging.info("Skipping gif creation.")

if __name__ == "__main__":
    main()
