# Melanoma Detection Case Study
> Oncological disease classification with convolutional neural networks.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information
The goal of this case study was to build multiclass classification models using custom convolutional neural networks classify photographs of malignant and benign oncological diseases and detect melanomas.

## Dataset
The dataset provided by the International Skin Imaging Collaboration contained 2239 training images and 118 test images belonging to 9 classes:
-   Actinic keratosis
-   Basal cell carcinoma
-   Dermatofibroma
-   Melanoma
-   Nevus
-   Pigmented benign keratosis
-   Seborrheic keratosis
-   Squamous cell carcinoma
-   Vascular lesion

The training dataset had significant class imbalance. The seborrheic keratosis class has the fewest number of elements, with 77 samples in the training dataset. In contrast, the pigmented benign keratosis class has 462 samples. The melanoma, basal cell carcinoma, and nevus classes are also overrepresented in the training dataset.

The ratio between the number of samples of the least and most represented classes is of six times. There is thus a significant class imbalance in the dataset.

## Methodology
Sequential Network and Residual Network models were constructed and trained using Keras and Tensorflow.

The training data was augmented with 500 generated images in each class. Generation of images involved the random rotation and flipping of images.  Images were rotated in multiples 90 degrees.  An additional rotation within the range of +-15 degrees was also applied, and the images were magnified to avoid introducing blank sections in the corners after rotation.  The training and validation data was separated using stratified sampling before augmentation was applied.

Sequential networks were created with 1,159,753 parameters.  These networks used five sequential blocks with two sequential 3x3 convolutional layers and ReLU activation functions each. Each convolutional layer employed 64 convolution kernals. Each block used a 2x2 max pooling layer, followed by a batch normalization layer. Dropouts were applied to all blocks. The final sequential model achieved a reasonable accuracy of 47% when classifying images in the test dataset.

A residual network model was created with bypass connections between processing blocks and lower layers. The model was based on the ResNet architecture. The model used twelve blocks, each with two sequential 3x3 convolutional layers and ReLU activation functions. The model achieved an accuracy of 42% when classifying images in the test dataset.

## Conclusions

All models showed signs of overfitting after approximately 25 epochs of training.  The overfitting did not appear to negatively impact the accuracy of the models when verified against the test dataset.

Rectifying the class imbalance in the training data through augmentation significantly increased the accuracy of the models when verified against the test dataset.

The sequential model outperformed the more complex and deeper residual network.  This can be attributed to the limited size of the training dataset which was likely insufficient to adequatly train the more complex network. Higher accuracy might be achievable when using transfer learning with a pretrained residual network.

While the purpose of this case study was to perform multiclass classification, binary classification may be more accurate for the identification of a single class -- melanomas in this case.

## Technologies Used
The following Python libraries were utilized:
- Keras
- Tensorflow
- Augmentor
- Numpy
- Pandas

## Acknowledgements
- The International Skin Imaging Collaboration

## Contact
Created by [@gertagenbag] - feel free to contact me!
