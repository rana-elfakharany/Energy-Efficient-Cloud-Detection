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
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score

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
        
    def Unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        num_classes = 1
      
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
      
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        b1 = BatchNormalization()(c1)
        r1 = ReLU()(b1)
        p1 = MaxPooling2D((2, 2))(r1)
      
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        b2 = BatchNormalization()(c2)
        r2 = ReLU()(b2)
        p2 = MaxPooling2D((2, 2))(r2)
      
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        b3 = BatchNormalization()(c3)
        r3 = ReLU()(b3)
        p3 = MaxPooling2D((2, 2))(r3)
      
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        b4 = BatchNormalization()(c4)
        r4 = ReLU()(b4)
        p4 = MaxPooling2D(pool_size=(2, 2))(r4)
      
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        b5 = BatchNormalization()(c5)
        r5 = ReLU()(b5)
        c5 = Dropout(0.3)(r5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
      
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = BatchNormalization()(u6)
        u6 = ReLU()(u6)
      
      
        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
        u7 = concatenate([u7, c3])
        u7 = BatchNormalization()(u7)
        u7 = ReLU()(u7)
      
      
        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
        u8 = concatenate([u8, c2])
        u8 = BatchNormalization()(u8)
        u8 = ReLU()(u8)
      
        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = BatchNormalization()(u9)
        u9 = ReLU()(u9)
      
      
        outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(u9)
        model = Model(inputs, outputs, name="Unet")
        return model
      
    K.clear_session()
        
    metrics = ['accuracy',
               tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
               tf.keras.metrics.AUC(), 
               tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
               sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    
    model = Unet(384, 384, 1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        CSVLogger(f"Unet_JARH_KFold/Results/CSVLogger/dataResUnet-fold{fold_number}.csv"), 
        TensorBoard(log_dir=f'Unet_JARH_KFold/Logs/logs-fold{fold_number}')
    ]
    
    results = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=30, callbacks=callbacks)
    
    model.save(f'Unet_JARH_KFold/Models/model-Unet-fold{fold_number}.hdf5')
    np.save(f'Unet_JARH_KFold/Results/results-fold{fold_number}.npy', results.history)
    
    df_result = pd.DataFrame(results.history)
    df_result.to_csv(f'Unet_JARH_KFold/Results/results-fold{fold_number}.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(f"Unet_JARH_KFold/Results/Loss/Loss-fold{fold_number}.png")
    
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["accuracy"], label="Accuracy")
    plt.plot(results.history["val_accuracy"], label="val_Accuracy")
    plt.plot(np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"Unet_JARH_KFold/Results/Accuracy/Accuracy-fold{fold_number}.png")
    
    gc.collect()

    print(f"Fold {fold_number} complete :)")    
