# Image-classifier-using-keras-and-tensorflow
🖼️ Image Classification with Keras & TensorFlow
📋 Project Overview
This project implements a Convolutional Neural Network (CNN) to automatically classify images into predefined categories. Built using the high-level Keras API on top of TensorFlow, this model processes raw pixel data to learn hierarchical spatial features, enabling it to distinguish between complex visual patterns.<br>

🧬 Model Architecture
The core of this project is a Sequential CNN designed to minimize spatial dimensions while increasing feature depth.

Input Layer: Rescaling of pixel values (0-255) to a normalized range (0-1).

Convolutional Blocks: Multiple Conv2D layers using ReLU activation to extract edges, textures, and shapes.

Pooling Layers: MaxPooling2D to reduce computational load and prevent overfitting.

Dense Head: A Flatten layer followed by Dense layers and a final Softmax (or Sigmoid) activation for class probability distribution.<br>

🚀 Key Features
Data Augmentation: Integrated tf.keras.layers.RandomFlip and RandomRotation to improve model generalization.

Performance Optimization: Utilizes the tf.data pipeline with AUTOTUNE for efficient data loading and prefetching.

Checkpointing: Automated saving of the "Best Model" based on validation loss during training.

Transfer Learning: (Optional/If used) Support for pre-trained backbones like ResNet50 or MobileNetV2.<br>

🛠️ Tech Stack
Framework: TensorFlow 2.x

High-level API: Keras

Tools: NumPy, Pandas (Metadata), Matplotlib (Visualization)<br>

📂 Dataset Structure
Organize your images in the following directory format for compatibility with image_dataset_from_directory:

Plaintext

/data
    /train
        /class_a
        /class_b
    /validation
        /class_a
        /class_b