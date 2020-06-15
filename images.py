import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage

def load_image(img_path):
    return cv2.imread(img_path)

def show_image(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)

def make_grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def convolution(img, conv_filter):
    return ndimage.convolve(img, conv_filter, mode = 'constant')

def normalize_image(img):
    img = img - np.min(img)
    img = img/np.max(img)
    return img

def load_normalize_and_grayscale(img_path):
    img = load_image(img_path)
    normalized_img = normalize_image(img.astype(np.float32))
    gray_img = make_grayscale(normalized_img)
    return img, normalized_img, gray_img

def split_image(img, array_size):
    splitted_img = []
    for x in range(0, img.shape[0]-array_size, array_size):
        for y in range(0, img.shape[1]-array_size, array_size):
            splitted_img.append(img[x:x+array_size, y:y+array_size])
    return splitted_img

def prepare_for_training(img_path, conv_filters):
    img, _, gray_img = load_normalize_and_grayscale(img_path)
    splitted_img = split_image(gray_img, 3)
    
    filtered_imgs = np.empty(shape=(gray_img.shape[0], gray_img.shape[1], 0))
    splitted_filtered_imgs = np.empty(shape=(len(splitted_img), 3, 3, 0))
    
    for conv_filter in conv_filters:
        filtered_img = convolution(gray_img, conv_filter)
        splitted_filtered_img = [convolution(x, conv_filter) for x in splitted_img]
        splitted_filtered_img = np.expand_dims(splitted_filtered_img, -1)
        
        filtered_imgs = np.append(filtered_imgs, filtered_img)
        splitted_filtered_imgs = np.append(splitted_filtered_imgs, splitted_filtered_img, axis=3)
        
    splitted_img = np.expand_dims(splitted_img, -1)
    
    return img, gray_img, filtered_imgs, splitted_img, splitted_filtered_imgs

def compare_images(img, original_img, out_image="compared_images.jpg", acceptance_error=0.2):
    out_image_folder = '/'.join(out_image.split('/')[:-1])
    if not os.path.exists(out_image_folder):
        os.makedirs(out_image_folder)
    
    combined_image = np.hstack((img, np.ones(shape=(img.shape[0], 10)), original_img))
    cv2.imwrite(out_image, 255.0*combined_image)

    similarity_array = np.isclose(img, original_img, atol=np.abs(np.max(original_img) - np.min(original_img))*(acceptance_error/100))
    perc_of_similarity = np.round(np.count_nonzero(similarity_array)/similarity_array.size*100, 2)

    return perc_of_similarity
