import cv2
import numpy as np
from scipy.linalg import hadamard

def perform_fft(image):
    # Convert the image to float32
    image_float32 = np.float32(image)

    # Split the image into color channels
    b, g, r = cv2.split(image_float32)

    # Perform DFT on each color channel
    dft_b = cv2.dft(b, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_g = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_r = cv2.dft(r, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Combine the results
    dft_combined = dft_b + dft_g + dft_r

    fft = np.fft.fftshift(dft_combined)

    magnitude_spectrum = cv2.magnitude(fft[:, :, 0], fft[:, :, 1])

    # Normalize the magnitude spectrum for saving
    magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    # Shift the zero frequency component to the center
    return magnitude_spectrum_normalized

def perform_dct(image):
    # Convert the image to float32
    image_float32 = np.float32(image)

    # Split the image into color channels
    b, g, r = cv2.split(image_float32)

    # Perform DCT on each color channel
    dct_b = cv2.dct(b)
    dct_g = cv2.dct(g)
    dct_r = cv2.dct(r)

    # Combine the results
    dct_combined = dct_b + dct_g + dct_r

    # Compute the magnitude spectrum (optional, similar to FFT)
    magnitude_spectrum = np.abs(dct_combined)

    # Normalize the magnitude spectrum for saving
    magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    return magnitude_spectrum_normalized

def perform_walsh(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray / 255.0
    rows, cols = img_gray.shape
    
    dimension = 2 ** int(np.ceil(np.log2(max(rows, cols))))
    
    resized_img = cv2.resize(img_normalized, (dimension, dimension))

    transform_matrix = hadamard(dimension)

    # Apply the Walsh-Hadamard transform
    transform_result = np.dot(np.dot(transform_matrix, resized_img), transform_matrix.T)

    return transform_result

def perform_haar(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transformed_image = np.copy(img_gray)
    rows, cols = transformed_image.shape

    # Transform along rows
    for i in range(rows):
        j = 0
        while j < cols:
            avg = (transformed_image[i, j] + transformed_image[i, j + 1]) / 2.0
            diff = (transformed_image[i, j] - transformed_image[i, j + 1]) / 2.0
            transformed_image[i, j] = avg
            transformed_image[i, j + 1] = diff
            j += 2

    # Transform along columns
    for j in range(cols):
        i = 0
        while i < rows:
            avg = (transformed_image[i, j] + transformed_image[i + 1, j]) / 2.0
            diff = (transformed_image[i, j] - transformed_image[i + 1, j]) / 2.0
            transformed_image[i, j] = avg
            transformed_image[i + 1, j] = diff
            i += 2

    return transformed_image


def perform_laplacian_of_gaussian(image, sigma=0):
    b, g, r = cv2.split(image)

    # Apply LoG filter to each channel
    b_filtered = cv2.Laplacian(cv2.GaussianBlur(b, (0, 0), sigma), cv2.CV_64F)
    g_filtered = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), sigma), cv2.CV_64F)
    r_filtered = cv2.Laplacian(cv2.GaussianBlur(r, (0, 0), sigma), cv2.CV_64F)

    # Combine channels
    result = cv2.merge([b_filtered, g_filtered, r_filtered])

    result_display = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    return result_display.astype(np.uint8)


def perform_histogram_equalization(image):
    equalized_image = cv2.equalizeHist(np.float32(image))

    return equalized_image


def perform_histogram_equalization(image):
    # Convert the image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Apply histogram equalization to the Y channel (luminance)
    yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])

    # Convert the image back to BGR color space
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return equalized_image