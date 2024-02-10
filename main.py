import os
from keras.preprocessing.image import ImageDataGenerator
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

dataset_path = "E:/Transfer Learning/cats_and_dogs_filtered.zip"

zip_object = zipfile.ZipFile(file=dataset_path, mode="r")
zip_object.extractall()
zip_object.close()
dataset_path_new = "E:/Transfer Learning/cats_and_dogs_filtered/"

# img_dir = dataset_path_new + '/train/'
# cat = 'cats/cat.0.jpg'
# dog = 'dogs/dog.0.jpg'
# images = [cat, dog]
# plt.figure(figsize=(20, 10))
#
# for i in range(2):
#     plt.subplot(1, 2, i + 1)
#     img = plt.imread(os.path.join(img_dir, images[i]))
#     plt.imshow(img)
#     plt.axis('off')
#     plt.colorbar()
#     print(img.shape)
#     print(
#         f"The dimensions of the image are {img.shape[0]} pixels width and {img.shape[1]} pixels height, three single color channel")
#     print(f"The maximum pixel value is {img.max()} and the minimum is {img.min():}")
#     print(f"The mean value of the pixels is {img.mean():.4f} and the standard deviation is {img.std():.4f}")
#     print()
#
#     for i in range(2):
#         rgb_img = plt.imread(os.path.join(img_dir, images[i]))
#         grayscale_img = color.rgb2gray(rgb_img)
#         sns.displot(grayscale_img.ravel(), kde=False)
#         plt.title('Distribution of Pixel Intensities in the Image')
#         plt.xlabel('Pixel Intensity')
#         plt.ylabel('Number Pixels in Image')
# cats and dogs
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")
data_gen_train = ImageDataGenerator(rescale=1 / 255.)
data_gen_valid = ImageDataGenerator(rescale=1 / 255.)
train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128, 128),
                                                     batch_size=128, class_mode="binary")
valid_generator = data_gen_valid.flow_from_directory(validation_dir, target_size=(128, 128),
                                                     batch_size=128, class_mode="binary")
IMG_SHAPE = (128, 128, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False,
                                               weights="imagenet")
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
prediction_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)
base_lr = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_lr), loss="binary_crossentropy",
              metrics=['accuracy'])
history = model.fit(train_generator, epochs=100, validation_data=valid_generator)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('Loss Curve')
plt.show()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title('Accuracy Curve')
plt.show()

base_model.trainable = True
len(base_model.layers)
fine_tune_at = 120
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_lr / 50), loss="binary_crossentropy",
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=10, validation_data=valid_generator)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('Loss Curve')
plt.show()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title('Accuracy Curve')
plt.show()
