# Import necessary libraries
from PIL import Image  # For image loading and manipulation
import numpy as np  # For numerical operations on arrays
from scipy.signal import convolve2d  # For performing 2D convolution
import matplotlib.pyplot as plt  # For plotting images

def load_image(image_path):
    """Load an image from a file path."""
    # Use PIL's Image.open to open the image file and convert it into a NumPy array for processing
    return np.array(Image.open(image_path))

def depthwise_convolution(image, kernel):
    """Perform depthwise convolution on an image."""
    # Ensure the input image has three dimensions [height, width, channels]
    # This is necessary because depthwise convolution is applied per channel
    if len(image.shape) < 3:
        raise ValueError("Depthwise convolution requires an image with at least 3 dimensions (height, width, channels).")
    
    # Extract the number of channels from the image shape
    channels = image.shape[2]
    # Prepare an output array with the same shape as the input image to store the result
    result = np.zeros_like(image)
    
    # Loop through each channel in the image
    for i in range(channels):
        # Apply the 2D convolution for the current channel
        # mode='same' ensures the output size matches the input size
        # boundary='wrap' handles the edges of the image
        result[:,:,i] = convolve2d(image[:,:,i], kernel, mode='same', boundary='wrap')
        
    return result

def pointwise_convolution(image, kernel):
    """Perform pointwise convolution on an image."""
    # The kernel size must match the number of channels in the image for pointwise convolution
    if len(kernel) != image.shape[2]:
        raise ValueError("For pointwise convolution, the kernel length must match the number of channels in the image.")
    
    # Reshape the kernel to be 3D for compatibility with the image data
    # This allows for multiplication across the channel dimension
    kernel = np.reshape(kernel, (1, 1, -1))
    # Multiply the image by the kernel and sum across the channels
    # keepdims=True keeps the third dimension after summation, ensuring the result remains 3D
    result = np.sum(image * kernel, axis=2, keepdims=True)
    
    return result

def apply_convolution(image_path, kernel, convolution_type='depthwise'):
    """Apply the specified type of convolution to an image."""
    # Load the image from the specified path
    image = load_image(image_path)
    
    # Apply the appropriate convolution type based on the parameter
    if convolution_type == 'depthwise':
        # If depthwise, call the depthwise convolution function
        result = depthwise_convolution(image, kernel)
    elif convolution_type == 'pointwise':
        # If pointwise, call the pointwise convolution function
        result = pointwise_convolution(image, kernel)
    else:
        # If the convolution type is unrecognized, raise an error
        raise ValueError("Unsupported convolution type. Please use 'depthwise' or 'pointwise'.")
    
    # Set up a figure for plotting the original and convoluted images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # Plotting the original image on the left
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')  # Hide axis for better visualization
    
    plt.subplot(1, 2, 2)  # Plotting the convoluted image on the right
    plt.title(f'{convolution_type.capitalize()} Convolution')
    # Ensure the result is within the valid image range [0, 255] and convert to uint8 for display
    plt.imshow(np.clip(result, 0, 255).astype(np.uint8))
    plt.axis('off')  # Hide axis for better visualization
    
    plt.show()  # Display the figure with the images



apply_convolution(r'D:\Study Files\NEU\Academic\INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]\handwritingnumbers_grayscale\cat.jpg', [[1, 0, -1], [1, 0, -1], [1, 0, -1]], 'depthwise')
apply_convolution(r'D:\Study Files\NEU\Academic\INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]\handwritingnumbers_grayscale\cat.jpg', [1, 0, -1], 'pointwise')


if __name__ == "__main__":
    apply_convolution(r'D:\Study Files\NEU\Academic\INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]\handwritingnumbers_grayscale\cat.jpg', [[1, 0, -1], [1, 0, -1], [1, 0, -1]], 'depthwise')
    apply_convolution(r'D:\Study Files\NEU\Academic\INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]\handwritingnumbers_grayscale\cat.jpg', [1, 0, -1], 'pointwise')

