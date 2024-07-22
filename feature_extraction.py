import os
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.applications.vgg16 import VGG16
from keras.src.models import Model
from keras.src.utils import get_file


def load_preprocessed_data():
    train_images = np.load('/data/train_images.npy')
    train_labels = np.load('/data/train_labels.npy')
    val_images = np.load('/data/val_images.npy')
    val_labels = np.load('/data/val_labels.npy')
    test_images = np.load('/data/test_images.npy')
    test_labels = np.load('/data/test_labels.npy')
    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def extract_features(images, batch_size=32):
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=vgg16_base.input, outputs=vgg16_base.output)

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = datagen.flow(images, batch_size=batch_size, shuffle=False)

    features = model.predict(generator, steps=len(images) // batch_size + 1, verbose=1)
    return features


def save_features(features, labels, prefix, data_path):
    np.save(os.path.join(data_path, f'data/{prefix}_features.npy'), features)
    np.save(os.path.join(data_path, f'data/{prefix}_labels.npy'), labels)


def main():
    data_path = 'C:/Users/aidam/PycharmProjects/pythonProject/'
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_preprocessed_data()

    train_features = extract_features(train_images)
    val_features = extract_features(val_images)
    test_features = extract_features(test_images)

    save_features(train_features, train_labels, 'train', data_path)
    save_features(val_features, val_labels, 'val', data_path)
    save_features(test_features, test_labels, 'test', data_path)
    print("Feature extraction complete!")


if __name__ == '__main__':
    main()
