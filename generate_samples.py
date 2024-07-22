import os
import numpy as np
import os
import cv2
# Set the backend
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from keras.src.applications.vgg16 import VGG16
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Model
matplotlib.use('Agg')


def save_sample_images(data_path, save_path, prefix, num_samples=5):
    images = np.load(os.path.join(data_path, f'{prefix}_images.npy'))
    labels = np.load(os.path.join(data_path, f'{prefix}_labels.npy'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(num_samples):
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f'{prefix}_sample_{i}.png'))
        plt.close()


def save_feature_extraction_samples(data_path, save_path, prefix, num_samples=5):
    images = np.load(os.path.join(data_path, f'{prefix}_images.npy'))

    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=vgg16_base.input, outputs=vgg16_base.output)

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = datagen.flow(images, batch_size=1, shuffle=False)
    features = model.predict(generator, steps=len(images))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(num_samples):
        feature_map = np.mean(features[i], axis=-1)
        plt.imshow(feature_map, cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f'feature_extraction_sample_{i}.png'))
        plt.close()


def save_synthetic_labels_samples(data_path, save_path, prefix, num_samples=5):
    labels = np.load(os.path.join(data_path, f'{prefix}_synthetic_labels.npy'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(num_samples):
        plt.bar(['Attention', 'Distraction'], labels[i])
        plt.title(f'Sample {i}')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(save_path, f'synthetic_label_sample_{i}.png'))
        plt.close()


def main():
    data_path = 'C:/Users/aidam/PycharmProjects/pythonProject/data'
    save_path = 'C:/Users/aidam/PycharmProjects/pythonProject/samples'

    # Save samples for preprocessing step
    save_sample_images(data_path, save_path, 'train')

    # Save samples for feature extraction step
    save_feature_extraction_samples(data_path, save_path, 'train')

    # Save samples for synthetic labels step
    save_synthetic_labels_samples(data_path, save_path, 'train')


if __name__ == '__main__':
    main()