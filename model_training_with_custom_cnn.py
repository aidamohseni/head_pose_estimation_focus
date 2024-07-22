import os
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def load_data(prefix, data_path):
    images = np.load(os.path.join(data_path, f'{prefix}_images.npy'))
    labels = np.load(os.path.join(data_path, f'{prefix}_synthetic_labels.npy'))
    return images, labels


def build_custom_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Output layer for attention and distraction
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    data_path = 'C:/Users/aidam/PycharmProjects/pythonProject/data'
    train_images, synthetic_train_labels = load_data('train', data_path)
    val_images, synthetic_val_labels = load_data('val', data_path)
    test_images, synthetic_test_labels = load_data('test', data_path)

    model = build_custom_cnn_model(train_images.shape[1:])

    checkpoint = ModelCheckpoint('best_custom_cnn_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.01,
                                 zoom_range=[0.9, 1.25], horizontal_flip=True, fill_mode='reflect')

    history = model.fit(datagen.flow(train_images, synthetic_train_labels, batch_size=32),
                        validation_data=(val_images, synthetic_val_labels),
                        epochs=100,
                        callbacks=[checkpoint, early_stopping])

    test_loss, test_accuracy = model.evaluate(test_images, synthetic_test_labels)
    print(f'Test Accuracy: {test_accuracy}')


if __name__ == '__main__':
    main()
