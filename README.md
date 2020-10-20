# Motivation

The aim of this project is to build a CNN-based algorithm that detects whether an image contains a dog or a human, and detects the dog breed that the image most resembles.

This is done through the use of three models:
* A pre-trained face detector is used from OpenCV for purposes of detecting humans.
* The ResNet-50 model (trained on the ImageNet dataset) is used to detect dogs in an image.
* A CNN model is built to predict dog breed. This is done in two ways: building a model from scratch, and using transfer learning on Xception model.

Note: This project forms part of Udacity's Data Scientist nanodegree.



# Libraries

* Python 3.7
* pandas
* seaborn
* scikit-learn
* keras
* opencv
* tqdm
* jupyter



# How to run the project

Please note that an html version of the main jupyter notebook is provided in case you have no need to run the code yourself, and merely want to read through the results. If you want the run the code yourself, follow the next steps.

## Download missing files

A number of files are too large to upload to GitHub, which is why they have to be downloaded and added separately.

1. Download the [dataset of dog images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). This should be unzipped and be placed inside `data/dogImages`.

2. Download the [dataset of human images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). This should be unzipped and placed in `data/lfw`.

3. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz). This should be placed in `bottleneck_features`.

4. Download the [Xception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz). This should be placed in `bottleneck_features`.


## Run

Once the above files have been downloaded, and the above libraries installed, open the dog_app.ipynb file with jupyter notebook, and run the code.




# Full list of files

* **bottleneck_features/**
  * **DogVGG16Data.npz**: This contains bottleneck features for the VGG16 model. Note: needs to be downloaded!
  * **DogXceptionData.npz**: This contains bottleneck features for the Xception model. Note: needs to be downloaded!

* **data/**
  * **dogImages/**: This directory contains train, validation, and test images of 133 different dog breeds for CNN training purposes. Note: needs to be downloaded!
  * **images/**: This directory contains images used for illustrative purposes within the main jupyter notebook file.
  * **lfw/**: This directory contains a dataset of human images. Note: needs to be downloaded!
  * **sample_images/**: This directory contains sample images that are used to assess the performance of the final algorithm in the jupyter notebook.

* **haarcascades/**
  * **haarcascade_frontalface_alt.xml**: This file contains a pre-trained face detector from OpenCV.

* **saved_models/**
  * **weights.best.from_scratch.hdf5**: This is the stored model weights for the best CNN built from scratch.
  * **weights.best.VGG16.hdf5**: This is the stored model weights for the example VGG16 model.
  * **weights.best.Xception.hdf5**: This is the stored model weights for the final Xception model.

* **dog_app.html**: This is an html version of the below jupyter notebook.
* **dog_app.ipynb**: This is the main jupyter notebook, where just about all of the actual code is.
* **extract_bottleneck_features.py**: This contains some functions for extracting bottleneck features that get imported in the boave jupyter notebook.



# Discussion

In this section I discuss the results of building a CNN from scratch, building a CNN through transfer learning, the final algorithm, as well as the points of improvement.

## CNN from scratch
There were a number of limitations and challenges that worked against this approach:
* Relatively small dataset (~8000 images).
* Large number of classes, with some classes having very minor differences (133).
* Development happening on CPU enforcing a practical limit on the size of the CNN, and the number of epochs that can be run.

Considering the above limitations, the final architecture's test accuracy of 12.3% (compared to the <1% accuracy of random guessing) is a rather good result.

This was achieved through following some general guidelines on how to build CNNs for image recognition. Primarily, through successive convolutional layers gradually increasing the number of nodes, with MaxPooling layers interspered to reduce output size; and the use of ReLu activation functions throughout.

Small gains on the accuracy were achieved through testing various parameters, with very small convolutional kernel_size achieving the best results.

However, large gains were achieved through the use of Image augmentation, which effectively makes up for the relatively small dataset. Images were augmented by shifting horizontally and vertically, flipping horizontally, and tilting the image up to 30 degrees. Flipping vertically and tilting the image by higher degrees was avoided, as this did not make sense for images of dogs.

## Transfer learning
As expected, taking an existing CNN trained on ImageNet and using transfer learning yielded much better results, with the final model achieving a test accuracy of 86.0%. The model that was chosen was Xception, largely arbitrarily. It is likely that other models could have yielded similar results.

As with the CNN built from scratch, image augmentation was used to increase performance of the model. However, considering the strong performance of this model to begin with, the performance boost from this technique was not as drastic as it was in the prior case.

## Final algorithm
The final algorithm is based on three models:
* OpenCV face detector for detecting humans.
* ResNet-50 model to detect dogs.
* Xception model to predict dog breed.

In practice, the Xception model feels like it is performing the strongest. This is because even when the model gets a dog breed wrong, it tends to predict a dog breed that strongly resembles the actual breed. And, in the case of human images, it is incredibly hard to judge what is or is not a good prediction, so the expectations of the model are rather low.

On the other hand, the face detector implementation feels like the weakest link. However, this has less to do with the objective accuracy of the model (which in testing was rather high), but rather the high expectations put on the model. While the Xception model has the benefit of being able to predict similar dog breeds even when it gets it wrong, and the fact that any dog breed can be seen as a "good" prediction for a human image, it is very obvious to a user when the face detector gets its output wrong. It can only output a binary yes/no, and if the model gets its prediction wrong, it is very clear to the user that it has done so.

## Improvements

Based on the discussion above, improvements should focus on the face detector model. It would first be important to more thoroughly test the current accuracy of the model, and it would be interesting to see whether alternative ready models would provide better performance (for instance, could the ResNet-50 model used for detecting dogs be used to detect humans?).

If the performance of the Xception model needed to be improved, this could be done in a few ways:
* Increase the size of the training dataset.
* Increase the data quality of the training dataset (some dogs are mis-classified).
* By developing on GPU, larger Dense layers could be experimented with, and the number of epochs run could also be increased.






# Acknowledgements

Thanks to Udacity for the datasets, provided bottleneck features, as well as the starter code.

All sample images are taken from Unsplash. Full image credits, in numbered order:
1. Joseph Gonzalez
2. Art Hauntington
3. Angel Luciano
4. Edson Torres
5. Jaycee Xie
6. Amber Kipp
