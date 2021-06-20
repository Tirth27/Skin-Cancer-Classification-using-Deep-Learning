from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras import backend as K

import albumentations as A
from ImageDataAugmentor.image_data_augmentor import ImageDataAugmentor

import utils

# Initialise the EfficientNet Model for transfer learning
def EffNet(input_size, num_classess, pretrained_model, lr_rate, print_trainable_layers = False, print_model_summary = False):
    # Get the EfficientNet Model
    base_model = pretrained_model(
        weights='imagenet',
        input_shape = input_size,
        include_top = False)
    
    # Keep the BatchNorm layer freeze, and unfreeze all other layers
    def unfreeze_model(model, print_trainable, print_summary):
        # unfreeze the layers while leaving BatchNorm layers frozen
        for layer in model.layers[:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        # Print trainable layer summary
        if print_trainable:
            for layer in model.layers:
                print(layer, layer.trainable)
        
        # Print Model summary
        if print_summary:
            base_model.summary()

    # Unfreeze the model
    unfreeze_model(base_model, print_trainable_layers, print_model_summary)

    # Add dense and output layer
    model = keras.Sequential()
    model.add(base_model)
    model.add(layers.Flatten(name='top_flatten'))
    model.add(layers.Dense(500, activation='relu', name='dense_500'))
    model.add(layers.Dense(256, activation='relu', name='dense_256'))
    model.add(layers.Dense(num_classess, activation='softmax', name='output_layer'))

    # Initialise the optimizer and compile the model
    optimizer = Adam(learning_rate = lr_rate)
    model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # print the FC layer summary
    if print_model_summary:
        model.summary() 

    return model 

# Fit the model on training and validation dataset and star the training process.
def train_model(model, train_generator, epoch, train_batch_size, validation_generator, validation_batch_size, train_step, valid_step,
callback):
    return model.fit(
      train_generator,
      epochs = epoch,
      batch_size = train_batch_size,
      validation_data = validation_generator,
      validation_batch_size = validation_batch_size,
      steps_per_epoch = train_step,
      validation_steps = valid_step,
      verbose = 1,
      callbacks = callback)

# Augment the dataset
def augment_images(image_size):
    transforms_train = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5),
        A.RandomBrightness(limit=0.2, p=0.5),
        A.RandomContrast(limit=0.2, p=0.5),
        A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0))
                ], p=0.7),
        A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3)
                ], p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(width=image_size, height=image_size),
        A.Cutout(max_h_size= int(image_size*0.375), max_w_size= int(image_size*0.375), num_holes=1, p=0.7),
        A.Normalize(),
    ])

    transforms_val = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize()
    ])

    transforms_test = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize()
    ])

    return transforms_train, transforms_val, transforms_test

# Generate the training and validation augmented dataset. 
def data_generator(seed, transforms_train, transforms_val, label, 
train_path, image_resize, train_batch_size, validation_batch_size):

    train_datagen = ImageDataAugmentor(
        augment = transforms_train,
        preprocess_input = None, 
        seed = seed,
        validation_split = 0.2) # Define validation split i.e 20% data is used for validation

    valid_datagen = ImageDataAugmentor(
            augment = transforms_val,
            preprocess_input = None, 
            seed = seed,
            validation_split = 0.2) # Define validation split i.e 20% data is used for validation

    # Flow training images using train_datagen generator
    train_generator = train_datagen.flow_from_dataframe(
            dataframe = label,  
            directory = train_path,
            x_col = 'image',
            y_col = 'diagnosis',
            target_size= image_resize, 
            batch_size = train_batch_size,
            subset = 'training',
            class_mode='categorical',
            validate_filenames = False)

    validation_generator = valid_datagen.flow_from_dataframe(
            dataframe = label, 
            directory = train_path,
            x_col = 'image',
            y_col = 'diagnosis',
            target_size= image_resize,
            batch_size = validation_batch_size,
            subset = 'validation',
            class_mode='categorical',
            validate_filenames = False)

    return train_generator, validation_generator