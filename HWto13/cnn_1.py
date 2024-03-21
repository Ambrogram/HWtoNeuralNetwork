import numpy as np

def convolve2d(image, kernel):
    """
    Perform a 2D convolution operation without padding and striding.
    
    Parameters:
    - image: 2D array of the input image.
    - kernel: 2D array of the filter.
    
    Returns:
    - Convoluted image as a 2D array.
    """
    # Determine the dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the shape of the output matrix
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize the output with zeros
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            # Element-wise multiplication and sum
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output

# Example neural network program segment

# Define an input image
image = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36]
])

# Define a filter (kernel)
kernel = np.array([
    [0, 1, 2],
    [2, 2, 0],
    [0, 1, 2]
])

# Perform the convolution
convolved_image = convolve2d(image, kernel)

print("Convolved Image:")
print(convolved_image)
