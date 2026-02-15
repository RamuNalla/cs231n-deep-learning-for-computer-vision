import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt

def apply_sobel(image_array):
    """
    Applies Sobel filters to an input numpy array (grayscale image).
    """
    # 1. Manually define the 3x3 Sobel Filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # 2. Apply convolution
    # mode='same' ensures the output matches input dimensions
    grad_x = signal.convolve2d(image_array, sobel_x, boundary='fill', mode='same')
    grad_y = signal.convolve2d(image_array, sobel_y, boundary='fill', mode='same')

    # 3. Calculate Magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return grad_x, grad_y, grad_magnitude

# --- TESTING CODE ---

# 1. Create a synthetic test image (100x100 pixels)
# A black background with a white square in the middle
test_img = np.zeros((100, 100))
test_img[25:75, 25:75] = 255  

# 2. Run the filter
gx, gy, magnitude = apply_sobel(test_img)

# 3. Visualization
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(test_img, cmap='gray')
axs[0].set_title('1. Original (Square)')

# We use np.absolute because gradients can be negative (transition from white to black)
axs[1].imshow(np.absolute(gx), cmap='gray')
axs[1].set_title('2. Vertical Edges (Gx)')

axs[2].imshow(np.absolute(gy), cmap='gray')
axs[2].set_title('3. Horizontal Edges (Gy)')

axs[3].imshow(magnitude, cmap='gray')
axs[3].set_title('4. Combined Magnitude')

for ax in axs:
    ax.axis('off') # Hide pixel coordinates for a cleaner look

plt.tight_layout()
plt.show()