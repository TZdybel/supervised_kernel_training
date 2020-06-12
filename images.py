import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def load_image(img_path):
    return cv2.imread(img_path)

def show_image(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)

def make_grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

# def filter_image_sobelx(img):
#     sobelx = ndimage.sobel(img, mode='constant', axis=0)
#     return sobelx

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
    if img.shape[0] % array_size != 0 or img.shape[1] % array_size != 0:
        raise AttributeError("Image needs to be divided by array size")
    splitted_img = []
    for x in range(0, img.shape[0], array_size):
        for y in range(0, img.shape[1], array_size):
            splitted_img.append(img[x:x+array_size, y:y+array_size])
    return splitted_img

def prepare_for_training(img_path, conv_filter):
    img, _, gray_img = load_normalize_and_grayscale(img_path)
    
    filtered_img = convolution(gray_img, conv_filter)

    splitted_img = split_image(gray_img, 3)
    splitted_filtered_img = [convolution(x, conv_filter) for x in splitted_img]

    splitted_img = np.expand_dims(splitted_img, -1)
    splitted_filtered_img = np.expand_dims(splitted_filtered_img, -1)
    
    return img, gray_img, filtered_img, splitted_img, splitted_filtered_img

def compare_images(img, original_img, out_image="compared_images.jpg"):
    combined_image = np.hstack((img, np.ones(shape=(img.shape[0], 10)), original_img))
    cv2.imwrite(out_image, 255.0*combined_image)
    
    differences = []
    for pix1, pix2 in zip(img.flatten(), original_img.flatten()):
        differences.append(np.abs(pix2 - pix1))
    avg_diff = np.round((np.average(differences)/(np.abs(np.max(original_img) - np.min(original_img)))) * 100, 4)

    return avg_diff
