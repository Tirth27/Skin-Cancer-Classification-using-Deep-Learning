from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
'''
# Expected Input shape for EfficientNet Model
    Base model	resolution
EfficientNetB0	224
EfficientNetB1	240
EfficientNetB2	260
EfficientNetB3	300
EfficientNetB4	380
EfficientNetB5	456
EfficientNetB6	528
EfficientNetB7	600
'''
# Initialise the Model Hyperparameter and training settings used to configure the model.
def model_parameter(selected_model):
    model_list = {
        "model2": {
            "backbone": EfficientNetB4,
            "target": 9,
            "resize": 380,
            "metadata": False,
            "initial_lr": 3e-5,
            "epochs": 15,
            'train_batch_size': 8,
            'validation_batch_size': 8,
            "savedModelByName": "Model2_EffB4_No_meta.h5",
            "saveFinalModelBy": "Model2",
            'log_by': "Model2_EffB4_No_meta.csv",
            'save_plot_name': 'Model2_EffB4_No_meta',
            'prediction_csv_name': 'Model2_EffB4_No_meta_prediction',
            'print_hyper_parameter': True,
            'input_image_size': 768,
            'print_trainable_layers': True, 
            'print_model_summary': False,
            'visualise_augmented_data': False
        },
        "model10": {
            "backbone": EfficientNetB5,
            "target": 9,
            "resize": 448,
            "metadata": False,
            "initial_lr": 3e-5,
            "epochs": 15,
            'input_image_size': 512,
            'train_batch_size': 4,
            'validation_batch_size': 4,
            "savedModelByName": "Model10_EffB5_No_meta.h5",
            "saveFinalModelBy": "Model10",
            'log_by': "Model10_EffB5_No_meta.csv",
            'save_plot_name': 'Model10_EffB5_No_meta',
            'prediction_csv_name': 'Model10_EffB5_No_meta_prediction',
            'print_hyper_parameter': True,
            'print_trainable_layers': False, 
            'print_model_summary': False,
            'visualise_augmented_data': False
        },
        "model12": {
            "backbone": EfficientNetB6,
            "target": 9,
            "resize": 528,
            "metadata": False,
            "initial_lr": 3e-5,
            "epochs": 15,
            'input_image_size': 768,
            'train_batch_size': 8,
            'validation_batch_size': 8,
            "savedModelByName": "Model12_EffB6_No_meta.h5",
            "saveFinalModelBy": "Model12",
            'log_by': "Model12_EffB6_No_meta.csv",
            'save_plot_name': 'Model12_EffB6_No_meta',
            'prediction_csv_name': 'Model12_EffB6_No_meta_prediction',
            'print_hyper_parameter': True,
            'print_trainable_layers': False, 
            'print_model_summary': False,
            'visualise_augmented_data': False
        },
        "model16": {
            "backbone": EfficientNetB7,
            "target": 9,
            "resize": 380,
            "metadata": False,
            "initial_lr": 1e-5,
            "epochs": 15,
            'input_image_size': 768,
            'train_batch_size': 4,
            'validation_batch_size': 4,
            "savedModelByName": "Model16_EffB7_No_meta.h5",
            "saveFinalModelBy": "Model16",
            'log_by': "Model16_EffB7_No_meta.csv",
            'save_plot_name': 'Model16_EffB7_No_meta',
            'prediction_csv_name': 'Model16_EffB7_No_meta_prediction',
            'print_hyper_parameter': True,
            'print_trainable_layers': False, 
            'print_model_summary': False,
            'visualise_augmented_data': False
        }
    } 

    return model_list[selected_model]
