# Load Libraries
from glob import glob
from functools import reduce
import os 
import sys
import yaml
import shutil
from tqdm import tqdm
import pandas as pd
import pprint

import utils
import pre_train
from model_param import model_parameter

from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

import albumentations as A
from ImageDataAugmentor.image_data_augmentor import ImageDataAugmentor

from azureml.core import Workspace, Dataset
from azureml.core.compute import ComputeTarget, ComputeInstance

if __name__ == '__main__':
    # Load Azure subscription Detail
    # Subscription detail for Instance one
    azure_auth_stream = open("Azure_outh_settings_INSTANCE_1.yml", 'r')
    # Subscription detail for Instance two
    #azure_auth_stream = open("Azure_outh_settings_INSTANCE_2.yml", 'r')

    azure_settings = yaml.load(azure_auth_stream, yaml.SafeLoader)

    ## Change working Directory 
    os.chdir('../')
    print("\nCurrent Working Directory: ", os.getcwd())

    # Azure subscription detail
    subscription_id = azure_settings['subscription_id']
    resource_group = azure_settings['resource_group']
    workspace_name = azure_settings['workspace_name']

    workspace = Workspace(subscription_id, resource_group, workspace_name)
    print("\nAzure Workspace Name: ", workspace.name)
    print("Azure Workspace Resource Group: ", workspace.resource_group)

    # Initialise Azure Instance
    try:
        instance = ComputeInstance(workspace = workspace, name = azure_settings['instance_name'])
        # Get Status
        print('Azure ML Instance is {}.\n'.format(instance.get_status().state))       
    except:
        print("An exception occurred while initialising the Azure ML Compute")
        sys.exit()

    # Select model to train
    # We have model2, model10, model16 and model12 model available
    model_config = 'model10'

    # Get the model parameters
    selected_model = model_parameter(model_config)

    # Define Model Log and Plot files
    log_path = "./runs/"
    os.makedirs(log_path, exist_ok = True)

    plot_path = "./plot/"
    os.makedirs(plot_path, exist_ok = True)

    ##################### Define path for Training and Testing Data
    train_path = "./{0}x{0}/".format(selected_model['input_image_size'])
    test_path = "./{0}x{0}_test/".format(selected_model['input_image_size'])

    save_model_path = "./saveModel/"
    os.makedirs(save_model_path, exist_ok = True)
    
    ##################### Get Training and Testing Labels from Azure Instance
    train_label = Dataset.get_by_name(workspace, name='train_2020_and_2019_with_9_Labels')
    test_label = Dataset.get_by_name(workspace, name='test_2020_no_PateintDetail')

    label = train_label.to_pandas_dataframe()
    test_csv = test_label.to_pandas_dataframe()

    # Append Image extension and file path to train and test CSV
    absolute_path_train = os.path.abspath(train_path)
    label = utils.append_path(label, absolute_path_train)

    absolute_path_test = os.path.abspath(test_path)
    test_csv = utils.append_path(test_csv, absolute_path_test)

    ##################### Hyper Parameter
    hyper_param = {
        'seed': 42,
        'image_size': selected_model['resize'], # resize image 
        'backbone_model': selected_model['backbone'], # Pretrained model name
        'early_stop': 10,
        'num_class': selected_model['target'],
        'train_batch_size': selected_model['train_batch_size'], # Train Batch Size
        'test_batch_size': 1, # Testing set batch size
        'validation_batch_size': selected_model['validation_batch_size'], # Validation Batch Size
        'epoch': selected_model['epochs'],
        'warmup_epoch': 1,
        'learning_rate_base': selected_model['initial_lr'], # Base learning rate after warmup.
        'warmup_learning_rate': selected_model['initial_lr'], # Warmup learning rate
        'training_sample_count': label.shape[0], # Number of training sample
        'save_model': selected_model['savedModelByName'], # save model name
        'save_final_model': selected_model['saveFinalModelBy'] # Save final trained model in Tensorflow Format
    }  

    image_resize = (hyper_param['image_size'], hyper_param['image_size'])
    image_shape = image_resize + (3, )
    # Total training steps in Warmup
    total_steps = int(hyper_param['epoch'] * hyper_param['training_sample_count'] / hyper_param['train_batch_size'])
    # Compute the number of warmup batches.
    warmup_steps = int(hyper_param['warmup_epoch'] * hyper_param['training_sample_count'] / hyper_param['train_batch_size'])

    # Print Hyper parameter
    if selected_model['print_hyper_parameter']:
        print("\n####################### Hyper Parameter #################################\n")
        pprint.pprint(hyper_param)
        
        print('\nImage Shape: {}'.format(image_shape))
        print('Total training steps in Warmup: {}'.format(total_steps))
        print('Number of Warmup Batch: {}\n'.format(warmup_steps))
        print("\nTrain Label shape: ", label.shape) 
        print("Test Label shape: ", test_csv.shape) 

    ########################### Train model
    # Create the Learning rate scheduler.
    warm_up_lr = utils.WarmUpCosineDecayScheduler(learning_rate_base = hyper_param['learning_rate_base'],
                                            total_steps = total_steps,
                                            warmup_learning_rate = hyper_param['warmup_learning_rate'],
                                            warmup_steps = warmup_steps,
                                            hold_base_rate_steps = 0)

    # Initialise Pre-train Model
    model = pre_train.EffNet(input_size = image_shape, num_classess = hyper_param['num_class'], \
        pretrained_model = hyper_param['backbone_model'], \
        lr_rate = hyper_param['learning_rate_base'], \
        print_trainable_layers = selected_model['print_trainable_layers'],\
        print_model_summary = selected_model['print_model_summary'])

    # Preprocess and Augment Image for train, test and validation set. 
    transform_train, transform_val, transform_test = \
        pre_train.augment_images(hyper_param['image_size'])

    # Prepare train, validation Generator
    train_generator, validation_generator = pre_train.data_generator(seed = hyper_param['seed'],\
        transforms_train = transform_train, transforms_val = transform_val, label = label, \
        train_path = train_path, image_resize = image_resize,  train_batch_size = selected_model['train_batch_size'], \
        validation_batch_size = selected_model['validation_batch_size'])

    # Visualise preprocess and augmented data
    if selected_model['visualise_augmented_data']:
        # Get Train set
        train_generator.show_data()
        # Get validation set
        validation_generator.show_data()
    
    ## Define Callbacks
    # Define Early Stopping on validation loss
    es = EarlyStopping(monitor='val_loss', mode = 'min', patience = hyper_param['early_stop'],\
        verbose = 1, restore_best_weights = True)

    # Save model after each epoch
    ck = ModelCheckpoint(save_model_path + hyper_param['save_model'], monitor='val_loss', \
        verbose = 1, save_best_only = False, save_weights_only= False, mode='auto')

    # Save logs to CSV
    # append=False -> overwrite existing file.
    logs = CSVLogger(log_path + selected_model['log_by'], separator=",", append=False)

    # Callback list
    call_backs = [warm_up_lr, ck, logs]

    # Get Train and validation step size
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

    start_training = input("Do you want to start training the model? [y]es OR [n]o: ")
    # Start the training process is the 'yes' input is received from the terminal 
    if start_training == 'y':
        print('\n\n---------------- Staring the Training Process... --------------- ')
        # Train the model
        history = pre_train.train_model(model = model, train_generator = train_generator, epoch = hyper_param['epoch'], \
            train_batch_size = hyper_param['train_batch_size'], validation_generator = validation_generator, \
            validation_batch_size = hyper_param['validation_batch_size'], train_step = STEP_SIZE_TRAIN, \
            valid_step = STEP_SIZE_VALID, callback = call_backs)
        print("\n ----------------- Model is trained --------------------------")
    else:
        print("Training is cancelled.....\nTerminating Python...")
        sys.exit()

    # Plot Training and validation loss
    print("\n------ Saving Training and Validation Plot --------")
    # Training and validation: accuracy & loss
    utils.save_plot(history = history, \
        save_dir = plot_path + selected_model['save_plot_name'])

    ############################ Predict on Testing Set
    test_datagen = ImageDataAugmentor(
            augment = transform_test,
            preprocess_input = None, 
            seed = hyper_param['seed'])
    
    # Define test generator
    test_generator = test_datagen.flow_from_dataframe(
        dataframe = test_csv,
        directory = test_path,
        x_col = 'image',
        target_size = image_resize,
        class_mode = None,
        batch_size = hyper_param['test_batch_size'],
        shuffle = False, 
        validate_filenames = False)

    # Get Test steps
    STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
    test_generator.reset()

    # predict on Testset
    print("\n------ Predicting on Testset --------")
    prediction = model.predict(test_generator, steps = STEP_SIZE_TEST, verbose = 1)
    predicted_class_indices = np.argmax(prediction, axis=1)

    # Map the predicted labels with their unique ids such as filenames.
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    # Save the prediction ot CSV File
    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename":filenames,
                            "Predictions":predictions})

    prediction_path = "./prediction/"
    os.makedirs(prediction_path, exist_ok = True)

    results.to_csv(prediction_path + selected_model['prediction_csv_name'] + ".csv", index=False)

    ##### Save Trained Model
    print("\n ------------ Saving the Trained model ------------------------------------")
    final_model_path = save_model_path + '{}/'.format(hyper_param['save_final_model'])
    os.makedirs(final_model_path, exist_ok = True)

    model.save(final_model_path, save_format="tf", include_optimizer = True)

    print("""\n---------------------- Completed Model Training ---------------------------\n
    ------------------------- Stopping the Azure Instance ------------------------""")
    # Stopping ComputeInstance will stop the billing meter and persist the state on the disk.
    # Available Quota will not be changed with this operation.
    instance.stop(wait_for_completion=True, show_output=True)
