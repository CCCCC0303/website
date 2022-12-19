from keras import models
from keras import layers
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD,Adam

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm 
import os

from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import gc

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it does something you should see a number as output

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))
    
    class_names = ['qualified','failed','feature']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)
IMAGE_SIZE = (416,416)

def load_data():
    datasets = ['cnn_train', 'cnn_test']#資料夾
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #cv讀照片，顏色默認為BGR，需轉為RGB，錯誤表示黑白或已轉
                image = cv2.resize(image, IMAGE_SIZE) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output
  
(train_images, train_labels), (test_images, test_labels) = load_data()
  
train_images, train_labels = shuffle(train_images , train_labels)
train_images = train_images /255.0
test_images =test_images /255.0

model = models.Sequential() 
model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (416,416,3))) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu')) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu')) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation = 'relu')) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation = 'relu')) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu')) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu')) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(512 ,activation = 'relu'))
model.add(layers.Dense(3, activation = 'softmax'))

model.compile(optimizer = 'adam', #避免局部最小化
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

reset_keras()

history = model.fit(train_images, train_labels, batch_size=5, epochs=200)

plt.title('train_loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot(history.history["loss"])

reset_keras()
'預測'
predictions = model.predict(test_images)     # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability
print(pred_labels)
print(predictions)

'混淆矩陣'
CM = confusion_matrix(test_labels, pred_labels)
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 
print(accuracy(CM))

'混淆矩陣視覺化，看錯誤'
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.show()

from keras.models import load_model
model.save('dnen2_model/keras_model.h5')

from keras import models
from keras import layers
from tensorflow.keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
import numpy as np

'存模型&讀模型'
from keras.models import load_model
model = load_model('dnen2_model/keras_model.h5') 

model.summary()
print(model.input)
print(model.output)

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.preprocessing import image

def image_prediction(path):
    IMAGE_PATH= path
    img=image.load_img(IMAGE_PATH,target_size=(416,416))
    img=image.img_to_array(img)
    plt.imshow(img/255.)
    predictions = model.predict(np.array([img]))
    if predictions[0,0] >= 1:
        print('qualified')
        print(predictions)
    elif predictions[0,1] >= 1:
        print('failed')
        print(predictions[0,1])
    elif predictions[0,2] >= 1:
        print("feature")
    elif predictions[0,:2] <= 1:
        print("no tire in the camera")

img_path = str(r'C:\Users\david\Downloads\wallhaven-95lx58.jpg')
img_path.replace('\\','/')
image_prediction(img_path)
