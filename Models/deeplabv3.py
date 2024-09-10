import os
import random
from datetime import datetime
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard

import segmentation_models as sm

from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
plt.style.use("ggplot")

print(tf.config.list_physical_devices('GPU'))
tf.debugging.set_log_device_placement(True)

with tf.device('/device:GPU:3'):  
    ## Seeding 
    seed = 2019
    random.seed = seed
    np.random.seed = seed
    tf.seed = seed
            
    w, h = 384, 384
    border = 5 
    
#     X = np.load('Data/X-384.npy') 
#     y = np.load('Data/y-384.npy')
#    
#    print(X.shape, y.shape) 
    
    fold_number = 5
    
    X_train = np.load(f'Data/Fold{fold_number}/X_train_fold{fold_number}.npy')
    X_val = np.load(f'Data/Fold{fold_number}/X_val_fold{fold_number}.npy')
    y_train = np.load(f'Data/Fold{fold_number}/y_train_fold{fold_number}.npy')
    y_val = np.load(f'Data/Fold{fold_number}/y_val_fold{fold_number}.npy')
    
    
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    gc.collect()

    from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Add, AveragePooling2D, UpSampling2D, Concatenate
    
    def conv_block(inputs, filters, kernel_size, strides=1, padding='same', activation='relu', dilation_rate=1):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x
    
    def depthwise_conv_block(inputs, kernel_size, strides=1, padding='same', activation='relu'):
        x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x
    
    def ASPP_module(inputs, depth=256, kernel_sizes=(1, 3, 3, 3)):
        aspp_layers = []
        for kernel_size in kernel_sizes:
            aspp_layers.append(conv_block(inputs, depth, 1, dilation_rate=1))
            aspp_layers.append(conv_block(inputs, depth, kernel_size, dilation_rate=kernel_size))
        x = Concatenate()(aspp_layers)
        return x
    
    def DeepLabV3(input_shape=(384, 384, 1), num_classes=1):
        inputs = Input(shape=input_shape)
       
        # Entry Flow
        x = conv_block(inputs, 32, 3, strides=2)  # size becomes 192x192
        x = conv_block(x, 64, 3)  # size remains 192x192
        x = conv_block(x, 64, 3, strides=2)  # size becomes 96x96
        x = conv_block(x, 128, 3)  # size remains 96x96
        x = conv_block(x, 128, 3, strides=2)  # size becomes 48x48
        x = conv_block(x, 256, 3)  # size remains 48x48
    
        # Middle Flow
        for _ in range(8):
            residual = x
            x = depthwise_conv_block(x, 3)
            x = conv_block(x, 256, 1, activation=None)
            x = Add()([residual, x])
    
        # Exit Flow
        x = ASPP_module(x)
        x = conv_block(x, 256, 1)
        x = UpSampling2D(size=(8, 8), interpolation='bilinear')(x)  # size becomes 384x384
        x = Conv2D(num_classes, 1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, x)
        return model
    
    # Create an instance of the DeepLabV3 model
    model = DeepLabV3(input_shape=(384, 384, 1), num_classes=1)
      
    from tensorflow.keras import backend as K
    from sklearn.metrics import f1_score
    
    K.clear_session()
        
    metrics = ['accuracy',
               tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
               tf.keras.metrics.AUC(), 
               tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
               sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        CSVLogger(f"DeeplabV3_KFold/Results/CSVLogger/dataResUnet-fold{fold_number}.csv"), 
        TensorBoard(log_dir=f'DeeplabV3_KFold/Logs/logs-fold{fold_number}')
    ]
    
    results = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=30, callbacks=callbacks)
    
    model.save(f'DeeplabV3_KFold/Models/model-Unet-fold{fold_number}.hdf5')
    np.save(f'DeeplabV3_KFold/Results/results-fold{fold_number}.npy', results.history)
    
    df_result = pd.DataFrame(results.history)
    df_result.to_csv(f'DeeplabV3_KFold/Results/results-fold{fold_number}.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(f"DeeplabV3_KFold/Results/Loss/Loss-fold{fold_number}.png")
    
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["accuracy"], label="Accuracy")
    plt.plot(results.history["val_accuracy"], label="val_Accuracy")
    plt.plot(np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"DeeplabV3_KFold/Results/Accuracy/Accuracy-fold{fold_number}.png")
    
    gc.collect()

    print(f"Fold {fold_number} complete :)")    
