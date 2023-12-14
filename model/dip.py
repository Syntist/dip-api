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

def perform_walsh_transform(image):
    gray_image = np.mean(image, axis=-1)
    N = len(gray_image)
    H = hadamard(N) / np.sqrt(N)
    return np.dot(H, gray_image)


def perform_laplacian_of_gaussian(image, kernel_size=5, sigma=0):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

    # Convert the result to 8-bit for display
    laplacian = np.uint8(np.absolute(laplacian))

    return laplacian


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