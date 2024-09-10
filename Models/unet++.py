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
    
    fold_number = 4
    
    X_train = np.load(f'Data/Fold{fold_number}/X_train_fold{fold_number}.npy')
    X_val = np.load(f'Data/Fold{fold_number}/X_val_fold{fold_number}.npy')
    y_train = np.load(f'Data/Fold{fold_number}/y_train_fold{fold_number}.npy')
    y_val = np.load(f'Data/Fold{fold_number}/y_val_fold{fold_number}.npy')
    
    
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    gc.collect()

    def conv2d(layer_input, filters, conv_layers=3):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        return d

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u
    
    def Unetpp(filters, output_channels, width, height, input_channels=1, conv_layers=3):

        inputs = Input(shape=(width, height, input_channels))
    
        conv00 = conv2d(inputs, filters, conv_layers=conv_layers)
        pool0 = MaxPooling2D((2, 2))(conv00)
    
        conv10 = conv2d(pool0, filters * 2, conv_layers=conv_layers)
        pool1 = MaxPooling2D((2, 2))(conv10)
    
        conv01 = deconv2d(conv10, filters)
        conv01 = concatenate([conv00, conv01])
        conv01 = conv2d(conv01, filters, conv_layers=conv_layers)
    
        conv20 = conv2d(pool1, filters * 4, conv_layers=conv_layers)
        pool2 = MaxPooling2D((2, 2))(conv20)
    
        conv11 = deconv2d(conv20, filters)
        conv11 = concatenate([conv10, conv11])
        conv11 = conv2d(conv11, filters, conv_layers=conv_layers)
    
        conv02 = deconv2d(conv11, filters)
        conv02 = concatenate([conv00, conv01, conv02])
        conv02 = conv2d(conv02, filters, conv_layers=conv_layers)
    
        conv30 = conv2d(pool2, filters * 8, conv_layers=conv_layers)
        pool3 = MaxPooling2D((2, 2))(conv30)
    
        conv21 = deconv2d(conv30, filters)
        conv21 = concatenate([conv20, conv21])
        conv21 = conv2d(conv21, filters, conv_layers=conv_layers)
    
        conv12 = deconv2d(conv21, filters)
        conv12 = concatenate([conv10, conv11, conv12])
        conv12 = conv2d(conv12, filters, conv_layers=conv_layers)
    
        conv03 = deconv2d(conv12, filters)
        conv03 = concatenate([conv00, conv01, conv02, conv03])
        conv03 = conv2d(conv03, filters, conv_layers=conv_layers)
    
        conv40 = conv2d(pool3, filters * 16)
    
        conv31 = deconv2d(conv40, filters * 8)
        conv31 = concatenate([conv31, conv30])
        conv31 = conv2d(conv31, 8 * filters, conv_layers=conv_layers)
    
        conv22 = deconv2d(conv31, filters* 4)
        conv22 = concatenate([conv22, conv20, conv21])
        conv22 = conv2d(conv22, 4 * filters, conv_layers=conv_layers)
    
        conv13 = deconv2d(conv22, filters * 2)
        conv13 = concatenate([conv13, conv10, conv11, conv12])
        conv13 = conv2d(conv13, 2 * filters, conv_layers=conv_layers)
    
        conv04 = deconv2d(conv13, filters)
        conv04 = concatenate([conv04, conv00, conv01, conv02, conv03], axis=3)
        conv04 = conv2d(conv04, filters, conv_layers=conv_layers)
    
        outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv04)
    
        model = Model(inputs=inputs, outputs=outputs)
    
        return model
    

    
    from tensorflow.keras import backend as K
    from sklearn.metrics import f1_score

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

    
    model = Unetpp(16, 1, w, h, input_channels=1, conv_layers=2)
      
    from tensorflow.keras import backend as K
    from sklearn.metrics import f1_score
    
    K.clear_session()
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        CSVLogger(f"Unet++_KFold/Results/CSVLogger/dataResUnet-fold{fold_number}.csv"), 
        TensorBoard(log_dir=f'Unet++_KFold/Logs/logs-fold{fold_number}')
    ]
    
    results = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=30, callbacks=callbacks)
    
    model.save(f'Unet++_KFold/Models/model-Unet-fold{fold_number}.hdf5')
    np.save(f'Unet++_KFold/Results/results-fold{fold_number}.npy', results.history)
    
    df_result = pd.DataFrame(results.history)
    df_result.to_csv(f'Unet++_KFold/Results/results-fold{fold_number}.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(f"Unet++_KFold/Results/Loss/Loss-fold{fold_number}.png")
    
    plt.figure(figsize=(15, 6))
    plt.title("Learning curve")
    plt.plot(results.history["accuracy"], label="Accuracy")
    plt.plot(results.history["val_accuracy"], label="val_Accuracy")
    plt.plot(np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"Unet++_KFold/Results/Accuracy/Accuracy-fold{fold_number}.png")
    
    gc.collect()

    print(f"Fold {fold_number} complete :)")    
