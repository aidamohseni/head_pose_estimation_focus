import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib

matplotlib.use('Agg')  # Use the Agg backend for non-interactive environments


def load_images_and_labels(dataset_dir, target_size=(224, 224), batch_size=100):
    images = []
    head_poses = []
    batch_images = []
    batch_head_poses = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                try:
                    head_pose = extract_pose_info(file)
                    image = cv2.imread(img_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, target_size)
                        batch_images.append(image)
                        batch_head_poses.append(head_pose)

                        if len(batch_images) >= batch_size:
                            images.append(np.array(batch_images, dtype=np.float32) / 255.0)
                            head_poses.append(np.array(batch_head_poses, dtype=np.float32))
                            batch_images, batch_head_poses = [], []

                    else:
                        print(f"Warning: {img_path} could not be read and will be skipped.")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    if batch_images:
        images.append(np.array(batch_images, dtype=np.float32) / 255.0)
        head_poses.append(np.array(batch_head_poses, dtype=np.float32))

    images = np.concatenate(images, axis=0)
    head_poses = np.concatenate(head_poses, axis=0)

    return images, head_poses


def extract_pose_info(filename):
    try:
        parts = filename.split('_')
        pitch = float(parts[-3].replace('P', ''))
        yaw = float(parts[-2].replace('V', ''))
        roll = float(parts[-1].replace('.jpg', '').replace('.png', '').replace('H', ''))
        return [pitch, yaw, roll]
    except Exception as e:
        raise ValueError(f"Filename {filename} does not match expected format. Error: {e}")


def visualize_samples(images, head_poses, sample_count=5, output_file='visualization.png'):
    plt.figure(figsize=(15, 5))
    for i in range(sample_count):
        plt.subplot(1, sample_count, i + 1)
        plt.imshow(images[i])
        plt.title(f"Head Pose: {head_poses[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def split_data(images, head_poses, test_size=0.2, val_size=0.25):
    train_images, test_images, train_labels, test_labels = train_test_split(images, head_poses, test_size=test_size,
                                                                            random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                          test_size=val_size, random_state=42)
    return train_images, val_images, test_images, train_labels, val_labels, test_labels


def augment_data(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=[0.9, 1.25],
        horizontal_flip=True,
        fill_mode='reflect'
    )
    augmented_images, augmented_labels = next(datagen.flow(train_images, train_labels, batch_size=len(train_images)))
    return augmented_images, augmented_labels


def save_preprocessed_data(train_images, val_images, test_images, train_labels, val_labels, test_labels):
    np.save('data/train_images.npy', train_images)
    np.save('data/train_labels.npy', train_labels)
    np.save('data/val_images.npy', val_images)
    np.save('data/val_labels.npy', val_labels)
    np.save('data/test_images.npy', test_images)
    np.save('data/test_labels.npy', test_labels)


def preprocess_and_save_data(dataset_dir):
    images, head_poses = load_images_and_labels(dataset_dir)
    visualize_samples(images, head_poses, output_file='initial_samples.png')
    train_images, val_images, test_images, train_labels, val_labels, test_labels = split_data(images, head_poses)
    augmented_train_images, augmented_train_labels = augment_data(train_images, train_labels)
    visualize_samples(augmented_train_images, augmented_train_labels, output_file='augmented_samples.png')
    save_preprocessed_data(augmented_train_images, val_images, test_images, augmented_train_labels, val_labels,
                           test_labels)
    print("\nPreprocessing Complete!")
