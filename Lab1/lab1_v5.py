import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import zipfile



zip_file_path = 'D:/Study Files/NEU/Academic/INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]/DogCat/train.zip'

# Opening the zip file using 'with' ensures it gets closed properly after use
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    filenames = zip_ref.namelist()
    # Now, filenames is a list of all the files in the zip archive
    
    
filenames = os.listdir('D:/Study Files/NEU/Academic/INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]/DogCat/train.zip')
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(str(1))
    else:
        categories.append(str(0))

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


df.head() , df.tail()


df['category'].value_counts()




# Assume 'df' is already created with 'filename' and 'category' columns

# Split the DataFrame into train and validation sets
train_data, valid_data = train_test_split(df, test_size=0.2, random_state=42)


def batch_generator(df, batch_size=32, target_size=(128, 128), shuffle=True):
    """
    Generator function for creating batches of images and labels on-the-fly.
    
    Args:
    - df: DataFrame containing the filenames and categories.
    - batch_size: Size of each batch.
    - target_size: Tuple specifying the image dimensions.
    - shuffle: Whether to shuffle the dataset at the beginning of each epoch.
    
    Yields:
    - A tuple (batch_images, batch_labels) for each batch.
    """
    num_samples = len(df)
    while True:  # Loop indefinitely
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data at the start of each epoch
        for offset in range(0, num_samples, batch_size):
            batch_samples = df.iloc[offset:offset+batch_size]
            images = np.zeros((len(batch_samples), *target_size, 3), dtype=np.float32)
            labels = np.zeros((len(batch_samples), 1), dtype=np.float32)
            for i, (filename, category) in enumerate(zip(batch_samples['filename'], batch_samples['category'])):
                img_path = os.path.join('/kaggle/working/train/', filename)
                img = load_img(img_path, target_size=target_size)
                images[i] = img_to_array(img) / 255.0  # Normalize images
                labels[i] = int(category)
            yield images, labels

# Create generators for training and validation sets
train_gen = batch_generator(train_data, batch_size=32)
valid_gen = batch_generator(valid_data, batch_size=32)

# Example usage of the generator
# for images, labels in train_gen:
#     # This is where you would feed the images and labels to your model for training
#     # For demonstration purposes, break after one step
#     break


