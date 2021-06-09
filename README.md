# Skin Cancer Classification using Deep Learning
# Abstract
In cancer, there are over 200 different forms. Out of 200, melanoma is the deadliest form of skin cancer. The diagnostic procedure for melanoma starts with clinical screening, followed by dermoscopic analysis and histopathological examination. Melanoma skin cancer is highly curable if it gets identified at the early stages. The first step of Melanoma skin cancer diagnosis is to conduct a visual examination of the skin's affected area. Dermatologists take the dermatoscopic images of the skin lesions by the high-speed camera, which have an accuracy of 65-80% in the melanoma diagnosis without any additional technical support. With further visual examination by cancer treatment specialists and dermatoscopic images, the overall prediction rate of melanoma diagnosis raised to 75-84% accuracy. The project aims to build an automated classification system based on image processing techniques to classify skin cancer using skin lesions images.

# Introduction and Background
Among all the skin cancer type, melanoma is the least common skin cancer, but it is responsible for 75% of death [SIIM-ISIC Melanoma Classification, 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification). Being a less common skin cancer type but is spread very quickly to other body parts if not diagnosed early. The International Skin Imaging Collaboration (ISIC) is facilitating skin images to reduce melanoma mortality. Melanoma can be cured if diagnosed and treated in the early stages. Digital skin lesion images can be used to make a teledermatology automated diagnosis system that can support clinical decision.

Currently, deep learning has revolutionised the future as it can solve complex problems. The motivation is to develop a solution that can help dermatologists better support their diagnostic accuracy by ensembling contextual images and patient-level information, reducing the variance of predictions from the model.

## The problem we tried to solve
The first step to identify whether the skin lesion is malignant or benign for a dermatologist is to do a skin biopsy. In the skin biopsy, the dermatologist takes some part of the skin lesion and examines it under the microscope. The current process takes almost a week or more, starting from getting a dermatologist appointment to getting a biopsy report. This project aims to shorten the current gap to just a couple of days by providing the predictive model using Computer-Aided Diagnosis (CAD). The approach uses CNN (Convolutional Neural Network) to classify nine types of skin cancer from outlier lesions images. This reduction of a gap has the opportunity to impact millions of people positively.

## Motivation
The overarching goal is to support the efforts to reduce the death caused by skin cancer. The primary motivation that drives the project is to use the advanced image classification technology for the well-being of the people. Computer vision has made good progress in machine learning and deep learning that are scalable across domains. With the help of this project, we want to reduce the gap between diagnosing and treatment. Successful completion of the project with higher precision on the dataset could better support the dermatological clinic work. The improved accuracy and efficiency of the model can aid to detect melanoma in the early stages and can help to reduce unnecessary biopsies.

## Application
We aim to make it accessible for everyone and leverage the existing model and improve the current system. To make it accessible to the public, we build an easy-to-use website. The user or dermatologist can upload the patient demographic information with the skin lesion image. With the image and patient demographic as input, the model will analyse the data and return the results within a split second. Keeping the broader demographic of people in the vision, we have also tried to develop the basic infographic page, which provides a generalised overview about melanoma and steps to use the online tool to get the results.

## Dataset
The project dataset is openly available on Kaggle [SIIM-ISIC Melanoma Classification, 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification). It consists of around forty-four thousand images from the same patient sampled over different weeks and stages. The dataset consists of images in various file format. The raw images are in DICOM (Digital Imaging and COmmunications in Medicine), containing patient metadata and skin lesion images. DICOM is a commonly used file format in medical imaging. Additionally, the dataset also includes images in TFRECORDS (TensorFlow Records) and JPEG format.

Furthermore, thirty-three thousand are in training set among the forty-four thousand images and around eleven thousand in the test set. However, our quick analysis found a significant class imbalance in the training dataset. Thirty-two thousand are labelled as benign (Not Cancerous) and only five hundred marked as malignant (Cancerous). That is, the training set contains only ±1.76% of malignant images (Figure 1). Along with the patient's images, the dataset also has a CSV file containing a detail about patient-level contextual information, which includes patient id, gender, patient age, location of benign/malignant site, and indicator of malignancy for the imaged lesion.

# Add Image Here

To overcome the issue of class imbalance, we planned to include data from the year 2018 [ISIC, 2018](https://challenge2018.isic-archive.com/) and 2019 [ISIC, 2019](https://challenge2019.isic-archive.com/) competition with our existing 2020 Kaggle competition [SIIM-ISIC Melanoma Classification, 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification). Out of 25k images in the 2019 competition, it has ten times (17.85%) more positive sample ratio, making the metrics more stable (Figure 2).

# Add Image Here

### Sample Images From Dataset

## Data Pre-Processing
Image crop things goes here.....

## Data Augmentation

# Overview of the Architecture

## CNN Architecture Design

### The reason behind choosing EfficientNet Architecture

## GUI Design

# Results and Evaluation

## Network Configurations

## Network Evaluation

# Model Evaluation and Deployment

## Limitations, Future Extension, and Improvements

# Conclusion

# References 

# Project Contribution
