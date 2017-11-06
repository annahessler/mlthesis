from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# print(os.listdir())
print('this is augment')
img = load_img('data/raw/riceRidge/perims/0731.tif') 
print(img) 
x = img_to_array(img)
x = img.reshape(img.shape + (1,1))
print('x shape is ', x.shape)
# x = img.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='data/raw', save_prefix='test', save_format='tif'):
    i += 1
    if i > 1:
        break  # otherwise the generator would loop indefinitely
