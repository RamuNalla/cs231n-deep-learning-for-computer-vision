import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt

def apply_sobel(image_path):
    # 1. Load image and convert to grayscale (L mode)
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # 2. Manually define the 3x3 Sobel Filters
    # Vertical Edge Detector
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # Horizontal Edge Detector
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # 3. Apply convolution using scipy
    # 'boundary=fill' with 'fillvalue=0' is equivalent to Zero Padding
    # 'mode=same' ensures the output is the same size as input
    grad_x = signal.convolve2d(img_array, sobel_x, boundary='fill', mode='same')
    grad_y = signal.convolve2d(img_array, sobel_y, boundary='fill', mode='same')

    # 4. Calculate the Magnitude of the gradient (combined edges)
    # G = sqrt(Gx^2 + Gy^2)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Plotting the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_array, cmap='gray')
    axs[0].set_title('Original Grayscale')
    
    axs[1].imshow(np.absolute(grad_x), cmap='gray')
    axs[1].set_title('Vertical Edges (Gx)')
    
    axs[2].imshow(grad_magnitude, cmap='gray')
    axs[2].set_title('Gradient Magnitude (All Edges)')
    
    plt.show()

# To run this, just provide a path to any image
# apply_sobel('your_image.jpg')