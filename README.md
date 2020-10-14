# Resnet50CatDog
Detecting cats and dogs is a simple deep learning task but the goal of this repo is Trasnfer Learning with Resnt50

# Contents:
    1. First downlaod the data (cat-dog) in [Kaggel](https://www.kaggle.com/c/dogs-vs-cats/data) 
    2. Second download the right weights for Resnet50 in [Kaggle](https://www.kaggle.com/keras/resnet50)
   
  
## Code Description
importing needed library : 
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import cv2
import os
```
importing Resnet50 model :
```python
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense
```

1. When you download the weight go and paste the weight in **.keras** folder in **C:/users/admin** *path*
2. then doing *Transfer learning* by writing **weights = 'imagenet' **
3. **include_top = False** is for removing Dense Layer
```python
model = Sequential()
# Sequential needed for add method
model.add(ResNet50(include_top = False, pooling = 'avg' ,weights = 'imagenet'))
model.add(Dense(2, activation='softmax'))
model.layers[0].trainable = False
```
1. Compiling model with **sgd** was SO MUCH better than **adam** optimizer.
2. In fact, with **adam** tha val_acc was %85 at most.
```python
from keras import optimizers
sgd = 'sgd'
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
```
1. Using **preprocess_input** for handling scaling by itselfs
2. Try to use more parameter in **train_data_generator** to decrease the overfit
```python
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

train_data_generator = ImageDataGenerator(preprocess_input, 
                                          horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

validation_data_generator = ImageDataGenerator(preprocess_input)

train_generator = train_data_generator.flow_from_directory(
    '/content/gdrive/My Drive/cat-dogs/New folder/train',
    target_size=(224, 224),
    batch_size = 32,
    class_mode='categorical'
)
validation_generator = validation_data_generator.flow_from_directory(
    '/content/gdrive/My Drive/cat-dogs/New folder/validation',
     target_size=(224, 224),
    batch_size = 32,
    class_mode='categorical'
)
```
And finally, **fitting** the model:
```python
fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=2000//32,
    validation_data=validation_generator,
    validation_steps=800//32,
    epochs=10
)
```
Saving the model : 
```python
model.save('/content/gdrive/My Drive/cat-dogs/New folder/cats_dogs_resnet.h5')
```

Loading the model:
```python
from tensorflow.keras.models import load_model
model = load_model('/content/gdrive/My Drive/cat-dogs/New folder/cats_dogs_resnet.h5')
```
Making a Dict for **Prediction**
```python
classes = {
   0: 'Dog',
   1: 'Cat'
}
```
1. Load a pic
2. Convert it to array
3. Normalizing 
4. predict_classes returns a **list** with **one NUMBER** using the NUMBER for indexing the Dict ABOVE and printing the class name
```python
img = image.load_img(img_path, target_size=(224, 224))
imshow(img)
x = image.img_to_array(img)
x /= 255
x = np.expand_dims(x , axis=0)
print(model.predict(x))
index = model.predict_classes(x)
for x in index:
    print(classes[x])
```
