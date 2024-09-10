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

    def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                  kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # second layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                  kernel_initializer = 'he_normal', padding = 'same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x
 
    def Unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
        """Function to define the UNET Model"""
        # Contracting Path
        c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)
        
        c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)
        
        c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)
        
        c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)
        
        c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
        
        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        
        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        
        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        
        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model
      
    from tensorflow.keras import backend as K
    from sklearn.metrics import f1_score
    
    K.clear_session()
        
    metrics = ['accuracy',
               tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
               tf.keras.metrics.AUC(), 
               tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
               sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    
    input_img = Input((h, w, 1), name='img')
    
    model = Unet(input_img)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        CSVLogger(f"Unet_KFold/Results/CSVLogger/dataResUnet-fold{fold_number}.csv"), 
        TensorBoard(log_dir=f'Unet_KFold/Logs/logs-fold{fold_number}')
    ]
    
    results = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=30, callbacks=callbacks)
    
    model.save(f'Unet_KFold/Models/model-Unet-fold{fold_number}.hdf5')
    np.save(f'Unet_KFold/Results/results-fold{fold_number}.npy', results.history)
    
    df_result = pd.DataFrame(results.history)
    df_result.to_csv(f'Unet_KFold/Results/results-fold{fold_number}.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(f"Unet_KFold/Results/Loss/Loss-fold{fold_number}.png")
    
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["accuracy"], label="Accuracy")
    plt.plot(results.history["val_accuracy"], label="val_Accuracy")
    plt.plot(np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"Unet_KFold/Results/Accuracy/Accuracy-fold{fold_number}.png")
    
    gc.collect()

    print(f"Fold {fold_number} complete :)")    
