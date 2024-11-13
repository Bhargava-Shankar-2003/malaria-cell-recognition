# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset

## Neural Network Model

![image](https://github.com/user-attachments/assets/df6de166-7c43-416c-8bfe-9ce56f4f8400)

## DESIGN STEPS

We begin by importing the necessary Python libraries, including TensorFlow for deep learning, data preprocessing tools, and visualization libraries.
To leverage the power of GPU acceleration, we configure TensorFlow to allow GPU processing, which can significantly speed up model training.
We load the dataset, consisting of cell images, and check their dimensions. Understanding the image dimensions is crucial for setting up the neural network architecture.
We create an image generator that performs data augmentation, including rotation, shifting, rescaling, and flipping. Data augmentation enhances the model's ability to generalize and recognize malaria-infected cells in various orientations and conditions.
We design a convolutional neural network (CNN) architecture consisting of convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with appropriate loss and optimization functions.
We split the dataset into training and testing sets, and then train the CNN model using the training data. The model learns to differentiate between parasitized and uninfected cells during this phase.
We visualize the training and validation loss to monitor the model's learning progress and detect potential overfitting or underfitting.
We evaluate the trained model's performance using the testing data, generating a classification report and confusion matrix to assess accuracy and potential misclassifications.
We demonstrate the model's practical use by randomly selecting and testing a new cell image for classification.

## PROGRAM

### Name: Bhargava S

### Register Number: 212221040029

```python
#!/usr/bin/env python
# coding: utf-8

# Name: Bhargava S
# 
# Register Number: 212221040029
# 

# In[1]:


import tensorflow as tf
# to share the GPU resources for multiple sessions
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix


# In[5]:


my_data_dir = "dataset/cell_images"
os.listdir(my_data_dir)


# In[6]:


test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'


# In[7]:


os.listdir(train_path)


# In[8]:


len(os.listdir(train_path+'/uninfected/'))


# In[9]:


len(os.listdir(train_path+'/parasitized/'))


# In[10]:


os.listdir(train_path+'/parasitized')[0]


# In[11]:


para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])


# In[12]:


print("Bhargava S 212221040029")
plt.imshow(para_img)


# In[23]:


# Checking the image dimensions
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)


# In[24]:


sns.jointplot(x=dim1,y=dim2)


# In[25]:


image_shape = (130,130,3)


# In[26]:


help(ImageDataGenerator)


# In[27]:


image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


# In[28]:


image_gen.flow_from_directory(train_path)


# In[29]:


image_gen.flow_from_directory(test_path)


# Name: Bhargava S
# 
# Register Number: 212221040029

# In[30]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[32]:


model.summary()


# In[33]:


batch_size = 16


# In[34]:


help(image_gen.flow_from_directory)


# In[35]:


train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')


# In[36]:


train_image_gen.batch_size


# In[37]:


len(train_image_gen.classes)


# In[38]:


train_image_gen.total_batches_seen


# In[39]:


test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)


# In[40]:


train_image_gen.class_indices


# In[44]:


results = model.fit(train_image_gen,epochs=2,
                              validation_data=test_image_gen
                             )


# In[46]:


losses = pd.DataFrame(model.history.history)


# In[47]:


losses[['loss','val_loss']].plot()


# In[48]:


model.metrics_names


# In[49]:


model.evaluate(test_image_gen)


# In[55]:


pred_probabilities = model.predict(test_image_gen)


# In[50]:


test_image_gen.classes


# In[56]:


predictions = pred_probabilities > 0.5


# In[57]:


print("Bhargava S 212221040029")
print(classification_report(test_image_gen.classes,predictions))


# In[60]:


print("Bhargava S 212221040029")
confusion_matrix(test_image_gen.classes,predictions)


# In[59]:


import random
import tensorflow as tf
list_dir=["UnInfected","parasitized"]
dir_=(list_dir[1])
para_img= imread(train_path+ '/'+dir_+'/'+ os.listdir(train_path+'/'+dir_)[random.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(para_img))
img = tf.image.resize(img,(128,128))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,128,128,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred
    else "Un Infected")+"\nActual Value: "+str(dir_))
plt.axis("off")
print("Bhargava S\n212221040029\n")
plt.imshow(img)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/2fc47012-557d-482d-a1b5-3b1f99eaca67)


### Classification Report

![Screenshot 2024-11-13 142737](https://github.com/user-attachments/assets/0d08242a-c4de-4e9d-b759-9263f657ca70)


### Confusion Matrix

![Screenshot 2024-11-13 142745](https://github.com/user-attachments/assets/45304bde-1b3e-4e6c-9b3d-57cbfa9fd7b6)


### New Sample Data Prediction

![Screenshot 2024-11-13 142727](https://github.com/user-attachments/assets/229759b9-d9d8-46c5-8e0c-c19013feedd6)


## RESULT
Thus, a deep neural network for Malaria infected cell recognition is developed and the performance is analyzed.
