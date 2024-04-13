import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
# ********************************************** Grayscale $ Binary **********************************************
# ****************************************************************************************************************


def grayscale_conversion(image):    # Convert the color image to grayscale
    if len(image.shape) == 3:  # If the image is colorful convert it to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
# ________________________________________________________________________________________________________________


def binary_conversion(image, threshold=127):    # Apply binary thresholding
    _, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return binary_img
# ************************************************* Down Scaling *************************************************
# ****************************************************************************************************************


def down_scaling(image, new_size=(256, 256)):
    original_height, original_width = image.shape   # Height and width for the original image
    new_height, new_width = new_size    # Height and width for the required image

    scale_y = original_height / new_height  # Proportion for height
    scale_x = original_width / new_width    # Proportion for width

    downscaled_image = np.zeros(new_size, dtype='uint8')    # Reserve an empty image with the required size
    for y in range(new_height):     # Fill the required image from the original
        for x in range(new_width):
            original_y = int(y * scale_y)
            original_x = int(x * scale_x)
            downscaled_image[y, x] = image[original_y, original_x]

    return downscaled_image
# *************************************************** Analysis ***************************************************
# ****************************************************************************************************************


def image_analysis(image):
    # Calculate mean(average)
    height, width = image.shape
    num_of_pixels = height * width
    total_sum = 0

    for j in range(height):
        for i in range(width):
            total_sum += image[j, i]        # Sum of all pixels intensities

    mean_value = total_sum / num_of_pixels      # The sum of all pixels intensities divided by their number
    # ____________________________________________________________________________________________________________
    # Calculate standard deviation
    total = 0
    for j in range(height):
        for i in range(width):
            deviation = image[j, i] - mean_value    # Subtract the mean from each score
            squared_deviation = deviation ** 2      # Square each deviation
            total += squared_deviation        # Add the squared deviations

    result = total / num_of_pixels      # Divide the total by the # of pixels
    std_value = math.sqrt(result)       # Take the square root from the result
    # ____________________________________________________________________________________________________________
    # Histogram
    hist = [0] * 256   # 256 levels, all set to 0 (0 - 255)

    # Iterate over each pixel in the image and update histogram
    for y in range(height):
        for x in range(width):
            intensity = image[y, x]
            hist[intensity] += 1
    # ____________________________________________________________________________________________________________
    # Calculate normalized histogram
    normalized = [0] * 256      # 256 levels, all set to 0 (0 - 255)
    for i in range(256):
        normalized[i] = hist[i] / num_of_pixels
    # ____________________________________________________________________________________________________________
    # Calculate entropy
    entropy_value = 0

    for probability in normalized:
        if probability != 0:
            entropy_value -= probability * np.log2(probability)
    # ____________________________________________________________________________________________________________
    # Cumulative histogram
    cumulative = [0] * 256      # 256 levels, all set to 0 (0 - 255)
    cumulative[0] = hist[0]
    for i in range(1, 256):        # Current level = current level + all previous levels
        cumulative[i] = cumulative[i - 1] + hist[i]
    # ____________________________________________________________________________________________________________
    return mean_value, std_value, entropy_value, hist, normalized, cumulative
# ********************************************* Contrast Enhancement *********************************************
# ****************************************************************************************************************


def contrast_enhancement(image):
    height, width = image.shape
    num_of_pixels = height * width
    mean, std, entropy, histogram, normalized_histogram, cumulative_histogram = image_analysis(image)

    for j in range(height):     # Perform histogram equalization on each pixel
        for i in range(width):
            intensity = image[j, i]
            equalized_intensity = int((cumulative_histogram[intensity] / num_of_pixels) * 255)
            image[j, i] = equalized_intensity

    return image
# ________________________________________________________________________________________________________________
# def enhancement(image):
#     height, width = image.shape
#     min_value = float('inf')
#     max_value = float('-inf')
#
#     for j in range(height):     # Fine the min and max in the image
#         for i in range(width):
#             min_value = min(min_value, image[j, i])
#             max_value = max(max_value, image[j, i])
#
#     pixel_range = max_value - min_value
#     c = pixel_range * 0.5
#
#     new_min_value = min_value - c
#     new_max_value = max_value + c
#     new_difference = new_max_value - new_min_value
#
#     for j in range(height):
#         for i in range(width):
#             image[j, i] = (((image[j, i] - min_value) / pixel_range) * new_difference) + new_min_value
#
#     return image
# ********************************************* Flipping and Blurring ********************************************
# ****************************************************************************************************************


def flip_image(image):         # Reflection
    height, width = image.shape
    flipped = np.zeros_like(image)        # Blank image

    for j in range(height):
        for i in range(width):
            flipped[j, i] = image[j, width - i - 1]

    return flipped
# ________________________________________________________________________________________________________________


def apply_blur(image):      # Blurring over all the picture
    height, width = image.shape
    blurred_img = image.copy()  # Make a copy of the original image
    blur_filter = np.ones((3, 3)) / 9       # 3x3 neighbours averaging filter

    # Iterate over each pixel of the image (except borders)
    for j in range(1, height - 1):      # Except the first and the last
        for i in range(1, width - 1):   # Except the first and the last
            neighborhood_sum = 0

            for dj in range(-1, 2):  # Iterate over neighborhood columns
                for di in range(-1, 2):  # Iterate over neighborhood rows
                    neighborhood_sum += image[j + dj, i + di] * blur_filter[dj + 1, di + 1]

            blurred_img[j, i] = int(neighborhood_sum)   # Assign the blurred pixel value
    return blurred_img
# ________________________________________________________________________________________________________________


def blurring_image(image):      # Blurring over borders of 80 pixels
    height, width = image.shape
    blurred_img = image.copy()  # Make a copy of the original image
    blur_filter = np.ones((3, 3)) / 9       # 3x3 neighbors averaging filter

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            if j < 80 or i < 80 or j > (height - 80) or i > (width - 80):
                neighborhood_sum = 0

                for dj in range(-1, 2):  # Iterate over neighborhood columns
                    for di in range(-1, 2):  # Iterate over neighborhood rows
                        neighborhood_sum += image[j + dj, i + di] * blur_filter[dj + 1, di + 1]
                blurred_img[j, i] = int(neighborhood_sum)  # Assign the blurred pixel value

    return blurred_img
# ************************************************* Negative Image ***********************************************
# ****************************************************************************************************************


def negative_image(image):      # Negative image
    img = 255 - image
    return img
# *************************************************** Crop Image *************************************************
# ****************************************************************************************************************


def crop_image(image, y, x, height, width):      # y & x coordinates, height & width
    img = image[y:y + height, x: x + width]
    return img
# ********************************************* Histogram Based Search *******************************************
# ****************************************************************************************************************
# def extract_vertical_strip(image, y, x, h, w):      # y & x coordinates, height & width
#     strip = np.zeros((h, w), dtype=np.uint8)
#
#     for j in range(h):
#         for i in range(w):
#             strip[j, i] = image[y + j, x + i]       # Fill the image in the strip
#     return strip
# ________________________________________________________________________________________________________________


def calculate_mean(arr):
    sum_arr = 0
    for elem in arr:    # Calculate the sum of all elements in the array
        sum_arr += elem

    mean_value = sum_arr / len(arr)   # Calculate the mean
    return mean_value
# ________________________________________________________________________________________________________________


def calculate_numerator(hist1, mean1, hist2, mean2):
    numerator = 0
    for x, y in zip(hist1, hist2):
        numerator += (x - mean1) * (y - mean2)
    return numerator
# ________________________________________________________________________________________________________________


def calculate_denominator(hist1, mean1, hist2, mean2):
    sum_sq_diff_hist1 = 0
    sum_sq_diff_hist2 = 0

    for x, y in zip(hist1, hist2):
        sum_sq_diff_hist1 += (x - mean1) ** 2
        sum_sq_diff_hist2 += (y - mean2) ** 2
    denominator = math.sqrt(sum_sq_diff_hist1) * math.sqrt(sum_sq_diff_hist2)
    return denominator
# ________________________________________________________________________________________________________________


def compare_histograms(img1, img2):   # Calculate correlation coefficient between histograms
    # hist1
    height1, width1 = img1.shape
    hist1 = [0] * 256       # 256 levels, all set to 0 (0 - 255)
    for y in range(height1):
        for x in range(width1):
            intensity = img1[y, x]
            hist1[intensity] += 1

    # hist2
    height2, width2 = img2.shape
    hist2 = [0] * 256       # 256 levels, all set to 0 (0 - 255)
    for y in range(height2):
        for x in range(width2):
            intensity = img2[y, x]
            hist2[intensity] += 1

    mean1 = calculate_mean(hist1)
    mean2 = calculate_mean(hist2)
    numerator = calculate_numerator(hist1, mean1, hist2, mean2)
    denominator = calculate_denominator(hist1, mean1, hist2, mean2)

    correlation = numerator / denominator if denominator != 0 else 0
    return correlation
# ________________________________________________________________________________________________________________


def find_vertical_strip(image, strip, threshold=0.95):
    height, width = image.shape
    strip_height, strip_width = strip.shape

    height_ratio = height / strip_height
    width_ratio = width / strip_width

    best_match_x = None
    best_match_y = None
    best_match_score = -1

    for y in range(height - strip_height + 1):  # Each possible values to start the height of the strip
        for x in range(width - strip_width + 1):    # Each possible values to start the width of the strip
            current_strip = crop_image(image, y, x, strip_height, strip_width)
            correlation = compare_histograms(strip, current_strip)
            if correlation > threshold and correlation > best_match_score:
                best_match_score = correlation
                best_match_x = x
                best_match_y = y

    best_match_x *= int(width_ratio)
    best_match_y *= int(height_ratio)
    height *= int(height_ratio)
    width *= int(width_ratio)

    return best_match_x, best_match_y, height, width
# *************************************************** Main *******************************************************
# ****************************************************************************************************************
# ****************************************************************************************************************
# ****************************************************************************************************************


# Load an image
original = cv2.imread('lena.jpg')
# ________________________________________________________________________________________________________________
# Apply histogram equalization
grayscale_image = grayscale_conversion(original)      # grayScale
binary_image = binary_conversion(grayscale_image)    # binary
cv2.imshow('Original Image', original)
cv2.imshow('Grayscale Image', grayscale_image)
cv2.imshow('Binary Image', binary_image)
# ________________________________________________________________________________________________________________
down_scaled_image = down_scaling(grayscale_image, new_size=(256, 256))   # downScaling
cv2.imshow('Downscaled Image', down_scaled_image)
# ________________________________________________________________________________________________________________
mean, std, entropy, histogram, normalized_histogram, cumulative_histogram = image_analysis(grayscale_image)
print("Mean:", mean)
print("Standard Deviation:", std)
print("Entropy:", entropy)
# ________________________________________________________________________________________________________________
plt.figure(figsize=(12, 8))
plt.plot(histogram, color='blue')
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(normalized_histogram, color='green')
plt.title('Normalized Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(cumulative_histogram, color='red')
plt.title('Cumulative Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()
# ________________________________________________________________________________________________________________
contrast_enhancement_image = contrast_enhancement(grayscale_image)
cv2.imshow('Enhanced Image', contrast_enhancement_image)
# s = enhancement(grayscale_image)
# cv2.imshow('Enhancement#2', s)
# ________________________________________________________________________________________________________________
flipped_image = flip_image(grayscale_image)
cv2.imshow('Flipped Image', flipped_image)
# ________________________________________________________________________________________________________________
blurred_image = apply_blur(grayscale_image)
image2 = blurring_image(grayscale_image)
cv2.imshow('Blured Image', blurred_image)
cv2.imshow('Blured2 Image', image2)
# ________________________________________________________________________________________________________________
negative_image = negative_image(grayscale_image)
cv2.imshow('Negative', negative_image)
# ________________________________________________________________________________________________________________
# cropped_image = crop_image(original, 0, 120, 400, 300)
# cv2.imshow('Cropped Image', cropped_image)
# ________________________________________________________________________________________________________________


x = 20
y = 20
h = 200
w = 200

s = crop_image(down_scaled_image, y, x, h, w)
cv2.imshow('Strip', s)

match_x, match_y, new_height, new_width = find_vertical_strip(down_scaled_image, s)

if match_x is not None and match_y is not None:
    print("Found the strip at x =", match_x, "and y =", match_y)
else:
    print("Sorry, strip is not found.")

cv2.rectangle(down_scaled_image, (match_y, match_x), (match_y + new_height, match_x + new_width), (0, 0, 255), 2)
cv2.imshow('Original Image with Strip', down_scaled_image)

cv2.waitKey(0)
cv2.destroyAllWindows()