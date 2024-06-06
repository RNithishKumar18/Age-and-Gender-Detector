# Age & Gender Detector

## Overview

This project demonstrates an application to detect age and gender from facial images using a Convolutional Neural Network (CNN). The project includes two main components:

1. **Model Training**: A Jupyter Notebook that trains a CNN model on the UTKFace dataset to predict age and gender.
2. **GUI Application**: A Python script providing a graphical user interface to upload an image and predict the age and gender of the person in the image.

## Dataset

The UTKFace dataset is used for training the model. It contains over 20,000 face images with corresponding age and gender labels. You can download the dataset from [Kaggle - UTKFace](https://www.kaggle.com/jangedoo/utkface-new).

## Contents

- `Age_Gender_Detector.ipynb`: Jupyter Notebook for training the age and gender prediction model.
-- `gui.py`: Python script for the GUI application.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies for the project.
- `Age_Sex_Detection.h5`: Pre-trained model weights (can be downloaded and placed in the root directory if not present).

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- pip (Python package installer)
- Jupyter Notebook (if you plan to run the `.ipynb` file)
- Virtual environment tool (optional, but recommended)

### Installation

1. **Clone the repository**:
   (https://github.com/RNithishKumar18/Age-and-Gender-Detector.git)
   cd age-gender-detector


# Model Training
If you want to train the model from scratch, follow these steps:

   Download the UTKFace dataset from Kaggle.

   Extract the dataset and place the images in a directory named UTKFace within the project folder.

- Run the Jupyter Notebook:
    jupyter notebook Age_Gender_Detector.ipynb

   Follow the steps in the notebook to train the model. 
   The trained model will be saved as Age_Sex_Detection.h5.

# Running the GUI Application
Ensure you have the pre-trained model file Age_Sex_Detection.h5 in the project directory. If not, download the file and place it in the root directory.

- Run the GUI application:
    python gui.py

This will open a GUI window where you can upload an image and get the predicted age and gender.

# Using the Application
- Upload an Image: Click the "Upload an Image" button and select an image file from your computer.
- Detect Age & Gender: After uploading the image, click the "Detect Image" button to see the predicted age and gender displayed on the screen.


# Project Structure
- Age_Gender_Detector.ipynb: Notebook for model training and evaluation.
- gui.py: GUI script for detecting age and gender from images.
- requirements.txt: List of required Python packages.
- Age_Sex_Detection.h5: Pre-trained model file.
- README.md: Project documentation (this file)


# Dependencies
The project requires the following Python packages:

tensorflow
opencv-python
numpy
matplotlib
Pillow
scikit-learn
tkinter


These can be installed using pip install -r requirements.txt.

# Acknowledgements

The UTKFace dataset provided by Kaggle.
