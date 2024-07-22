# head_pose_estimation_focus
This repository contains the implementation of a project aimed at head pose estimation using a custom Convolutional Neural Network (CNN). The project utilizes the Columbia Gaze dataset, involving steps from data preprocessing, creating synthetic labels for attention and distraction, to training a custom CNN model.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Head Pose Estimation Project</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        img { max-width: 100%; height: auto; display: block; margin-bottom: 20px; }
        .image-container { display: flex; flex-wrap: wrap; }
        .image-container img { margin: 10px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>Head Pose Estimation Project</h1>
    <p>This project aims to estimate head poses (attention and distraction) using a custom Convolutional Neural Network (CNN). The dataset used is the Columbia Gaze dataset. The project involves the following steps:</p>
    <ol>
        <li>Data Preprocessing</li>
        <li>Feature Extraction</li>
        <li>Creating Synthetic Labels</li>
        <li>Training a Custom CNN Model</li>
    </ol>
    <h2>Data Preprocessing</h2>
    <p>The data preprocessing step involves loading the images and extracting the head pose information from the filenames. The images are resized to 224x224 pixels and normalized. Here are some sample images from the training set:</p>
    <div class="image-container">
        <img src="samples/train_sample_0.png" alt="Training Sample 0">
        <img src="samples/train_sample_1.png" alt="Training Sample 1">
        <img src="samples/train_sample_2.png" alt="Training Sample 2">
        <img src="samples/train_sample_3.png" alt="Training Sample 3">
        <img src="samples/train_sample_4.png" alt="Training Sample 4">
    </div>

    <h2>Feature Extraction</h2>
    <p>Feature extraction is performed using a pre-trained VGG16 model. The model is loaded with the ImageNet weights, and the feature maps are extracted from the convolutional layers. These features are then used as input for the custom CNN model.</p>

    <h3>Understanding VGG16</h3>
    <p>The VGG16 model is a convolutional neural network architecture developed by the Visual Geometry Group at the University of Oxford. It is composed of 16 layers, including 13 convolutional layers and 3 fully connected layers. The model uses small 3x3 filters and max-pooling layers to reduce the spatial dimensions of the feature maps. VGG16 is widely used for image classification and object detection tasks due to its depth and ability to capture complex features. By using the pre-trained VGG16 model, we leverage its learned features from the ImageNet dataset, which helps in achieving better performance with less training data.</p>
    <p>For a detailed explanation of the VGG16 model, you can refer to the following sources:</p>
    <ul>
        <li><a href="https://neurohive.io/en/popular-networks/vgg16/" target="_blank">VGG16 - Convolutional Network for Classification and Detection</a></li>
        <li><a href="https://builtin.com/data-science/vgg16-keras" target="_blank">Beginners Guide to VGG16 Implementation in Keras</a></li>
        <li><a href="https://www.geeksforgeeks.org/vgg-16-cnn-model/" target="_blank">VGG16 | CNN model - GeeksforGeeks</a></li>
    </ul>

    <h2>Creating Synthetic Labels</h2>
    <p>Synthetic labels are generated based on the head pose information. If the pitch, yaw, and roll values are within a certain threshold, the label is marked as attention (focused). Otherwise, it is marked as distraction (not focused).</p>

    <h2>Training the Custom CNN Model</h2>
    <p>The custom CNN model is trained using the extracted features and synthetic labels. The model architecture consists of multiple convolutional layers, max-pooling layers, and dense layers with dropout for regularization. The model is trained with data augmentation techniques to improve generalization.</p>

    <h2>Results</h2>
    <p>The trained model is evaluated on the test set, and the test accuracy is reported. The model's performance indicates its ability to accurately classify head poses as attention or distraction.</p>

    <h2>Conclusion</h2>
    <p>This project demonstrates the use of deep learning techniques for head pose estimation. The custom CNN model, trained with augmented data and synthetic labels, shows promising results in classifying head poses based on the Columbia Gaze dataset.</p>
</body>
</html>

