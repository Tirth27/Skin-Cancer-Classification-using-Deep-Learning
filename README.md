# Skin Cancer Classification using Deep Learning
# Abstract
In cancer, there are over 200 different forms. Out of 200, melanoma is the deadliest form of skin cancer. The diagnostic procedure for melanoma starts with clinical screening, followed by dermoscopic analysis and histopathological examination. Melanoma skin cancer is highly curable if it gets identified at the early stages. The first step of Melanoma skin cancer diagnosis is to conduct a visual examination of the skin's affected area. Dermatologists take the dermatoscopic images of the skin lesions by the high-speed camera, which have an accuracy of 65-80% in the melanoma diagnosis without any additional technical support. With further visual examination by cancer treatment specialists and dermatoscopic images, the overall prediction rate of melanoma diagnosis raised to 75-84% accuracy. The project aims to build an automated classification system based on image processing techniques to classify skin cancer using skin lesions images.

# Introduction and Background
Among all the skin cancer type, melanoma is the least common skin cancer, but it is responsible for 75% of death [SIIM-ISIC Melanoma Classification, 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification). Being a less common skin cancer type but is spread very quickly to other body parts if not diagnosed early. The International Skin Imaging Collaboration (ISIC) is facilitating skin images to reduce melanoma mortality. Melanoma can be cured if diagnosed and treated in the early stages. Digital skin lesion images can be used to make a teledermatology automated diagnosis system that can support clinical decision.

Currently, deep learning has revolutionised the future as it can solve complex problems. The motivation is to develop a solution that can help dermatologists better support their diagnostic accuracy by ensembling contextual images and patient-level information, reducing the variance of predictions from the model.

## The problem we tried to solve
The first step to identify whether the skin lesion is malignant or benign for a dermatologist is to do a skin biopsy. In the skin biopsy, the dermatologist takes some part of the skin lesion and examines it under the microscope. The current process takes almost a week or more, starting from getting a dermatologist appointment to getting a biopsy report. This project aims to shorten the current gap to just a couple of days by providing the predictive model using **Computer-Aided Diagnosis (CAD)**. The approach uses CNN (Convolutional Neural Network) to classify nine types of skin cancer from outlier lesions images. This reduction of a gap has the opportunity to impact millions of people positively.

## Motivation
The overarching goal is to support the efforts to reduce the death caused by skin cancer. The primary motivation that drives the project is to use the advanced image classification technology for the well-being of the people. Computer vision has made good progress in machine learning and deep learning that are scalable across domains. With the help of this project, we want to reduce the gap between diagnosing and treatment. Successful completion of the project with higher precision on the dataset could better support the dermatological clinic work. The improved accuracy and efficiency of the model can aid to detect melanoma in the early stages and can help to reduce unnecessary biopsies.

## Application
We aim to make it accessible for everyone and leverage the existing model and improve the current system. To make it accessible to the public, we build an easy-to-use website. The user or dermatologist can upload the patient demographic information with the skin lesion image. With the image and patient demographic as input, the model will analyse the data and return the results within a split second. Keeping the broader demographic of people in the vision, we have also tried to develop the basic infographic page, which provides a generalised overview about melanoma and steps to use the online tool to get the results.

## Dataset
The project dataset is openly available on Kaggle [SIIM-ISIC Melanoma Classification, 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification). It consists of around forty-four thousand images from the same patient sampled over different weeks and stages. The dataset consists of images in various file format. The raw images are in DICOM (Digital Imaging and COmmunications in Medicine), containing patient metadata and skin lesion images. DICOM is a commonly used file format in medical imaging. Additionally, the dataset also includes images in TFRECORDS (TensorFlow Records) and JPEG format.

Furthermore, thirty-three thousand are in training set among the forty-four thousand images and around eleven thousand in the test set. However, our quick analysis found a significant class imbalance in the training dataset. Thirty-two thousand are labelled as benign (Not Cancerous) and only five hundred marked as malignant (Cancerous). That is, the training set contains only ±1.76% of malignant images (Figure 1). Along with the patient's images, the dataset also has a CSV file containing a detail about patient-level contextual information, which includes patient id, gender, patient age, location of benign/malignant site, and indicator of malignancy for the imaged lesion.

# Add Class Imbalance Image Here. Improve the Visualisation and Labels

To overcome the issue of class imbalance, we planned to include data from the year 2018 [ISIC, 2018](https://challenge2018.isic-archive.com/) and 2019 [ISIC, 2019](https://challenge2019.isic-archive.com/) competition with our existing 2020 Kaggle competition [SIIM-ISIC Melanoma Classification, 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification). Out of 25k images in the 2019 competition, it has ten times (17.85%) more positive sample ratio, making the metrics more stable (Figure 2).

# Add Training and testing images in 2020, 2019 and 2018 competition Image Here. Add Table rather then image. 

### Sample Images From Dataset
Figure 3, **ISIC_0015719** is labelled as benign melanoma in the dataset.

# Not Cancerous Image add few more sample

Figure 4, **ISIC_7079349** is labelled as malignant melanoma in the dataset.

# Figure 4: Cancerous Image add few more sample

## Data Pre-Processing

Add about label cleaning things

## Data Augmentation
In a small size dataset, image augmentation is required to avoid overfitting the training dataset. After data aggregation, we have around 46k images in the training set. The dataset contains significant class imbalance, with most of the classes have an **"Unknown"** category (Table 1). We have defined our augmentation pipeline to deal with the class imbalance. The augmentation that helps to improve the prediction accuracy of the model is selected. The selected augmentation are as follows:
1. **Transpose**: A spatial level transformation that transposes image by swapping rows and columns.
2. **Flip**: A spatial level transformation that flip image either/both horizontally and/or vertically. Images are randomly flipped either horizontally or vertically to make the model more robust.
3. **Rotate**: A spatial level transformation that randomly turns images for uniform distribution. Random rotation allows the model to become invariant to the object orientation.
4. **RandomBrightness**: A pixel-level transformation that randomly changes the brightness of the image. As in real life, we do not have object under perfect lighting conditions and this augmentation help to mimic real-life scenarios.
5. **RandomContrast**: A pixel-level transformation that randomly changes the contrast of the input image. As in real life, we do not have object under perfect lighting conditions and this augmentation help to mimic real-life scenarios.
6. **MotionBlur**: A pixel-level transformation that applies motion blur using a random-sized kernel.
7. **MedianBlur**: A pixel-level transformation that blurs input image using a median filter.
8. **GaussianBlur**: A pixel-level transformation that blurs input image using a gaussian filter.
9. **GaussNoise**: A pixel-level transformation that applies Gaussian noise to the image. This augmentation will simulate the measurement noise while taking the images
10. **OpticalDistortion**: Optical distortion is also known as Lens error. It mimics the lens distortion effect.
11. **GridDistortion**: An image warping technique driven by mapping between equivalent families of curves or edges arranged in a grid structure.
12. **ElasticTransform**: A pixel-level transformation that divides the images into multiple grids and, based on edge displacement, the grid will be distorted. This transform helps the network to have a better understanding of edges while training.
13. **CLAHE**: A pixel-level transformation that applies Contrast Limited Adaptive Histogram Equalization to the input image. This augmentation improves the contrast of the images.
14. **HueSaturationValue**: A pixel-level transformation that randomly changes hue, saturation and value of the input image.
15. **ShiftScaleRotate**: A spatial level transformation that randomly applies affine transforms: translate, scale and rotate the input. The allow scale and rotate the image by certain angles
16. **Cutout**: A spatial level transformation that does a rectangular cut in the image. This transformation helps the network to focus on the different areas in the images.

Figure 5 illustrates the before and after augmented image. The augmentation is applied to only the training set while just normalising the validation and testing dataset.

# Figure 5 Training set augmentation Add more sample images

After the data pre-processing and data augmentation, we have around 46,425 images in the training set, 11,606 images in the validation set and 10,875 images in the testing set. The training set is divided into an 80/20 ratio where 80% is used for training and 20% as a validation set.

# Overview of the Architecture
The project contains two flow diagrams. Figure 6 shows the model training pipeline, while Figure 7 shows the web UI flow. The first step after downloading the data is to clean and combine the data (Figure 6). The missing values in the patient demographic are imputed with the average values as the ratio of missing values is less than 5% in the overall dataset. The provided skin lesion images are of higher resolution, and it is not ideal for training the network on the high-resolution images (Figure 3 and 4). In the data pre-processing steps, all images are cropped into 768x786 and 512x512 resolution to reduce random noise on the edges of the image.

The data cleaning and pre-processing step are performed on all the dataset obtained from the 2020, 2019 and 2018 competition. Also, the image labels are reconciled and combined into a single training CSV file. The augmentation is performed on the fly during the model training process to reduce the storage space and improve efficiency. During the model training part, **Nth** images are read from the training folder and augmentation is performed on the CPU while the EfficientNet is loaded in the GPU. Augmentation is performed on the CPU, and training on GPU help to reduce the training time (Figure 6).

After each epoch, we check the validation accuracy of the model. If the validation accuracy does not increase after 15 epochs, the training process is stopped, and the best model weights are saved for the prediction (Figure 6). The prediction is performed on the test set, and results are stored in the CSV file. Along with the model weights, all diagnostic information for the model is stored locally.

# Figure 6 Model training flow diagram.

The web UI contains five pages, of which four of them are used to explain the project and how to use the proposed CAD system (Figure 7). The inference page named **"Our Solution"** is where the inference is made on the skin lesion images. All the validation is performed on the client-side to reduce the server overload. If the inserted information is not correct, then an error notification popup is shown; any user can easily understand that. Validated data is passed onto the server, where inference is performed by Onnx network, and response is return in the JSON format. On the website, we have configured the JQuery, which listen for server response. As soon as the response is return, the information is populated in a graphical format. Plus, for user convenience, we provide the functionality in which a user can generate the report in PDF format (Figure 7).

# Figure 7 Web UI flow diagram

## CNN Architecture Design
The project aims to classify skin cancer using skin lesions images. To achieve higher accuracy and results on the classification task, we have used various EfficientNet models. Transfer learning is applied to the EfficientNet models. We have unfrozen all the layer except BatchNormalization to stop the BatchNormalization layer from updating its means and variance statistics. If we train the BatchNormalisation layer, it will destroy what the model has learned, and accuracy will significantly reduce.

### The reason behind choosing EfficientNet Architecture
We have explored several different pre-trained models like VGG-16, VGG-19, ResNet-50 and ResNet-200. But the issue with these pre-train models is that they are depth scaled, meaning they are scaled up or down by adding or removing layers. For example, ResNet-50 is scaled up to ResNet-200 by adding more layers. Theoretically, adding layers to the network help to improve the performance, but it does not follow the same when implemented practically. As more layers are added to the network, it will face a vanishing gradient problem.

CNN's can be scaled with three dimensions: depth (d), width (w), and resolution (r). Depth scaling is adding or removing layers in the network. Width scaling makes the network wider, while resolution scaling increases or decreases image resolution passed in the CNN.

# Figure 8, Scaling dimension (Mingxing & Quoc, 2019)

Scaling width, depth and resolution improve network accuracy. However, it quickly saturates as the network is scaled only in a single dimension. From Figure 9, we can see that the baseline network (Figure 8) saturates at 80% when scaled in a single dimension.

# Figure 9, Accuracy saturation when scaling on a single dimension (Mingxing & Quoc, 2019)

EfficientNet used compound scaling (Figure 8), which uniformly scales the network's width, depth, and resolution. Among the different EfficientNet, EfficientNetB0 is the baseline network obtained by doing **Neural Architecture Search (NAS)**. EfficientNetB1 to B7 is built upon the baseline network having a different value of compound scaling.
**We have chosen to use EfficientNet B4, B5 and B7 as these model achieved start-of-the-art 84.4% top-1/ 97.1% top 5 accuracies (Mingxing & Quoc, 2019) on the ImageNet competition.**

## GUI Design
To tackle the challenge of identifying skin cancer from skin lesions, we have to build a predictive model for **Computer-Aided Diagnosis (CAD)**. Taking the skin lesions image and patient demographic information as input, we have developed a prototype web application that can help dermatologists interpret skin lesion images.

The web GUI consists of five main pages, of which four of them are used to explain the benefit of using the tool and way to reduce the death caused by skin cancer. The inference page named **"Our Solution"** is where the inference is performed using ensemble methodology.

The main page introduces the user to the approach we have chosen to scale across the domain where we merge the deep learning technology with the health care sector. Also, the main pages have four main sections (Figure 10, 11, 12 and 13). We have added button on the navigation bar for user convenience, which takes the user to the specified section.

# Figure 10 Main page (Section one)

We introduce the end-user to the melanoma and its severity in section two (Figure 11). Section two provides a generalised introduction of melanoma that the user can easily understand. Plus, we have provided a **"Explore"** button that redirects the user to the **"Info"** page (Figure 12). The information page provides in-depth information on the severity of skin cancer with its symptoms. The information page (Figure 12) is designed to keep the curious user in mind who wants to understand the problem profoundly.

# Figure 11 Main page (section two)

# Figure 12 Info Page

Once the user is familiar with skin cancer, we took the user to section three (Figure 13), showing how deep learning can help dermatologist in their clinical work. When the user clicks on the **"Explore"** button, they are redirected to the **"Tools"** page. The tools page will make the user familiar with deep learning and how it can help to reduce the death caused by melanoma skin cancer.

# Figure 13 Main page (Section three)

# Figure 14 Tools Page

In the last section of the main page (Figure 15), we introduce our CAD system. When **"Explore Our Solution"** is click, it will bring the end-user to the **"Our Solution"** page. The **"Our Solution"** page is where the inference of the skin lesion image is performed (Figure 16). The minimal materialised design is chosen, which looks attractive and encourage end-user to use the tool repeatedly.

The **"Our Solution"** page contains two main things. Firstly, a user needs to add the patient detail under the **"Fill Patient Detail"** section for which the inference is performed. Then a user needs to upload the skin lesion image. The validation is performed on the client-side using JQuery, and it will not allow the end-user to submit the detail until all the information is valid. The validation is performed on the client-side to reduce the server load.

# Figure 15 Main page (section four)

# Figure 16 Our Solution Page (Before Patient Details and Image Upload)

The validated information is sent to the server on the **"Upload"** button click where the network is ready to the server (Figure 17). The optimised network analyses the image, returning the inference to the client (Figure 18). The inference is automatically populated in the interactive bar graph (Figure 18). The bar graph is easy to read, and it infers the chances of having skin cancer and its type. The information that the end-user has inserted into the **"Fill Patient Detail"** section (Figure 17) is automatically populated in the inference section (Figure 18) for users' convenience. Also, we have provided the functionality to generate the report that can be stored locally for later use (Figure 19) just by click on the **"Generate PDF"** button. The PDF report includes the end-user information with the network prediction (Figure 19).

Moreover, we have also thought about patient privacy, and for the same reason, none of the patient demographic and skin lesion images are stored on the server. The server received the patient skin lesion image and performed the inference without storing it on the server.

# Figure 17 Our Solution Page (After Patient Details and image Upload)

# Figure 18 Our Solution Page (Network inference)

# Figure 19 Generated PDF report

Lastly, we have created an **"About Us"** page (Figure 20). The **"About Us"** page shows the core value of individual team members and their effort to deliver the end product.

# Figure 20 About Us Page

# Results and Evaluation
The model evaluation and performance on the test and validation images are as follows:

## Network Configurations
We have used ensemble terminology to train diverse models and take the average probability ranks of the models to get the final prediction. The model configuration is as follows:

1. **Backbone Pre-trained CNN Model**: *Efficient Net B4, B5 and B7*. We have chosen to use the B4, B5 and B7 variant of the efficient net over B0 as they have achieved higher accuracy on ImageNet competition.
2. **Targets**: All the model is trained on nine categories (Table 1).

| **Label** | **Name**                                                                             |
|-----------|--------------------------------------------------------------------------------------|
| **Mel**   | Melanoma                                                                             |
| **AK**    | Actinic keratosis                                                                    |
| **UNK**   | Unknown                                                                              |
| **VASC**      | Vascular lesion                                                                      |
| **BKL**       | Benign keratosis (solar lentigo / seborrheic keratosis/lichen planus-like keratosis) |
| **NV**        | Melanocytic nevus                                                                    |
| **BCC**       | Basal cell carcinoma                                                                 |
| **DF**       | Dermatofibroma                                                                       |
| **SCC**       | Squamous cell carcinoma                                                              |
Table 1, Label Name

3. **Original images are cropped** to *68x768* and *512x512* pixels. To reduce the random noise and black border on the edge of the images (Figure 2)
4. **Resized image input sizes** to *380x380* and *448x448* pixels. The images are resized to lower resolution due to GPU memory constraints. Otherwise, it was planned to load the images with the original cropped image pixels (Table 3).
5. **Cosine Decay learning rate** is set to *3e-5* and *1e-5* with *1* **Warmup epoch**. Along with the pre-trained model, we are using Cosine decay with a warmup learning rate scheduler. Warmup strategy gradually increases the learning rate from zero to the initial learning rate during initial **Nth** epochs or **m** batches. Cosine decay is used in conjunction with the warmup learning rate scheduler to decrease the initial learning rate value steadily. Cosine decay is used rather than exponential or steps decay. It reduces the learning rate slowly at the start and end while falling linearly in the middle—cosine decay help to improve the training process (Figure 21).

## Network Evaluation

# Model Evaluation and Deployment

## Limitations, Future Extension, and Improvements

# Conclusion

# References 

# Project Contribution
