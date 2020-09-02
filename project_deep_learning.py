#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import numpy as np # linear algebra
import pandas as pd 


# In[3]:


import keras
import matplotlib.pyplot as plt
from glob import glob 
from keras.models import Sequential 
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D,Input,SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
import cv2
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,confusion_matrix,classification_report


# In[ ]:


path_test = '/content/drive/My Drive/Massive data storage and retreival project/test'
path_train = '/content/drive/My Drive/Massive data storage and retreival project/train'
path_val = '/content/drive/My Drive/Massive data storage and retreival project/val'


# In[ ]:


#transforming the training images by rescaling, adding shear, zooming and flipping
train_gen = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip=True)

val_gen = ImageDataGenerator(rescale=1./255)


# In[ ]:


#create batch of images for training, test and validation sets
train_batch = train_gen.flow_from_directory(path_train,
                                            target_size = (224, 224),
                                            classes = ["NORMAL", "PNEUMONIA"],
                                            class_mode = "categorical")
val_batch = val_gen.flow_from_directory(path_val,
                                        target_size = (224, 224),
                                        classes = ["NORMAL", "PNEUMONIA"],
                                        class_mode = "categorical")
test_batch = val_gen.flow_from_directory(path_test,
                                         target_size = (224, 224),
                                         classes = ["NORMAL", "PNEUMONIA"],
                                         class_mode = "categorical")

print(train_batch.image_shape)


# In[ ]:


#building model
def build_model():
    input_img = Input(shape=train_batch.image_shape, name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=input_img, outputs=x)
    
    return model


# In[ ]:


model= build_model()
model.summary()


# In[ ]:


batch_size = 16
epochs = 10
early_stop = EarlyStopping(patience=25,
                           verbose = 2,
                           monitor='val_loss',
                           mode='auto')

checkpoint = ModelCheckpoint(
    filepath='best_model',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    verbose = 1)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=5,
    verbose=1, 
    mode='auto',
    min_delta=0.0001, 
    cooldown=1, 
    min_lr=0.0001
)

model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=Adam(lr=0.0001))

#fitting model on training data
history = model.fit_generator(epochs=epochs,
                              callbacks=[early_stop,checkpoint,reduce],
                              shuffle=True,
                              validation_data=val_batch,
                              generator=train_batch,
                              steps_per_epoch=500,
                              validation_steps=10,
                              verbose=2)


# In[ ]:


def create_plots(history):
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# In[ ]:


create_plots(history)


# In[ ]:


original_test_label=[]
images=[]

test_normal=Path('/content/drive/My Drive/Massive data storage and retreival project/test/NORMAL') 
normal = test_normal.glob('*.jpeg')
for i in normal:
    img = cv2.imread(str(i))
#     print("normal",img)
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        img = cv2.resize(img, (224,224))
    except Exception as e:
        print(str(e))
    images.append(img)
    label = to_categorical(0, num_classes=2)
    original_test_label.append(label)

test_pneumonia = Path("/content/drive/My Drive/Massive data storage and retreival project/test/PNEUMONIA")
pneumonia = test_pneumonia.glob('*.jpeg')
for i in pneumonia:
    img = cv2.imread(str(i))
#     print("pneumonia",img)
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        img = cv2.resize(img, (224,224))
    except Exception as e:
        print(str(e))
    images.append(img)
    label = to_categorical(1, num_classes=2)
    original_test_label.append(label)    

    
images = np.array(images)
original_test_label = np.array(original_test_label)
print(original_test_label.shape)


orig_test_labels = np.argmax(original_test_label, axis=-1)


# In[ ]:


p = model.predict(images, batch_size=16)
preds = np.argmax(p, axis=-1)
print(preds.shape)


# In[ ]:


#Accuracy and loss of model
test_loss, test_score = model.evaluate_generator(test_batch,steps=100)
print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)


# In[ ]:




