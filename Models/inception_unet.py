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

with tf.device('/device:GPU:2'):  
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

    def InceptionModule(inputs, filters):
        tower0 = Conv2D(filters, (1, 1), padding='same')(inputs)
        tower0 = BatchNormalization()(tower0)
        tower0 = Activation('relu')(tower0)
        
        tower1 = Conv2D(filters, (1, 1), padding='same')(inputs)
        tower1 = BatchNormalization()(tower1)
        tower1 = Activation('relu')(tower1)
        tower1 = Conv2D(filters, (3, 3), padding='same')(tower1)
        tower1 = BatchNormalization()(tower1)
        tower1 = Activation('relu')(tower1)
  
        tower2 = Conv2D(filters, (1, 1), padding='same')(inputs)
        tower2 = BatchNormalization()(tower2)
        tower2 = Activation('relu')(tower2)
        tower2 = Conv2D(filters, (3, 3), padding='same')(tower2)
        tower2 = Conv2D(filters, (3, 3), padding='same')(tower2)
        tower2 = BatchNormalization()(tower2)
        tower2 = Activation('relu')(tower2)
  
        tower3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        tower3 = Conv2D(filters, (1, 1), padding='same')(tower3)
        tower3 = BatchNormalization()(tower3)
        tower3 = Activation('relu')(tower3)
  
        inception_module = concatenate([tower0, tower1, tower2, tower3], axis=3)
  
        return inception_module
    
    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
  
        return u
    
    def inception_unet(filters, output_channels, width=None, height=None, input_channels=1):
    
        inputs = Input(shape=(width, height, input_channels))
      
        conv1 = InceptionModule(inputs, filters)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
      
        conv2 = InceptionModule(pool1, filters * 2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
      
        conv3 = InceptionModule(pool2, filters * 4)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
      
        conv4 = InceptionModule(pool3, filters * 8)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
      
        conv5 = InceptionModule(pool4, filters * 16)
      
        up6 = deconv2d(conv5, filters * 8)
        up6 = InceptionModule(up6, filters * 8)
        merge6 = concatenate([conv4, up6], axis=3)
      
        up7 = deconv2d(merge6, filters * 4)
        up7 = InceptionModule(up7, filters * 4)
        merge7 = concatenate([conv3, up7], axis=3)
      
        up8 = deconv2d(merge7, filters * 2)
        up8 = InceptionModule(up8, filters * 2)
        merge8 = concatenate([conv2, up8], axis=3)
      
        up9 = deconv2d(merge8, filters)
        up9 = InceptionModule(up9, filters)
        merge9 = concatenate([conv1, up9], axis=3)
      
        outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(merge9)
      
        model = Model(inputs=inputs, outputs=outputs)
      
        return model
      
    from tensorflow.keras import backend as K
    from sklearn.metrics import f1_score

    K.clear_session()
    
    def f1_metric(y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
    
        f1 = 2 * precision * recall / (precision + recall + K.epsilon())
        return K.mean(f1)
    
    metrics = ['accuracy',
               tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
               tf.keras.metrics.AUC(), 
               tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
               sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    model = inception_unet(16, 1, w, h, input_channels=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        CSVLogger(f"Inception_Unet_KFold/Results/CSVLogger/dataResUnet-fold{fold_number}.csv"), 
        TensorBoard(log_dir=f'Inception_Unet_KFold/Logs/logs-fold{fold_number}')
    ]
    
    results = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=30, callbacks=callbacks)
    
    model.save(f'Inception_Unet_KFold/Models/model-Unet-fold{fold_number}.hdf5')
    np.save(f'Inception_Unet_KFold/Results/results-fold{fold_number}.npy', results.history)
    
    df_result = pd.DataFrame(results.history)
    df_result.to_csv(f'Inception_Unet_KFold/Results/results-fold{fold_number}.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(f"Inception_Unet_KFold/Results/Loss/Loss-fold{fold_number}.png")
    
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["accuracy"], label="Accuracy")
    plt.plot(results.history["val_accuracy"], label="val_Accuracy")
    plt.plot(np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"Inception_Unet_KFold/Results/Accuracy/Accuracy-fold{fold_number}.png")
    
    gc.collect()

    print(f"Fold {fold_number} complete :)")    
