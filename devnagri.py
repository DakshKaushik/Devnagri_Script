#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


classes = os.listdir("./Data/Train")
print(classes)


# In[ ]:


X = []
for c,i in enumerate(classes):
    for j in os.listdir("./Data/Train/"+i):
        img = cv2.imread("./Data/Train/"+i+"/"+j)
        X.append(img)


# In[ ]:


Y = []
for i in range(0,len(X)):
    Y.append(i//1700)
Y = np.array(Y)
X = np.array(X)
Y.reshape(-1)


# In[ ]:


from sklearn.utils import shuffle
X, Y = shuffle(X, Y)


# In[ ]:


num = 2
plt.imshow(X[num])
plt.xlabel(classes[Y[num]])


# In[ ]:


from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.05, random_state=0)


# In[ ]:


train_y.reshape(-1)
val_y.reshape(-1)
NUM_CLASSES = 46


# In[ ]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq='batch')


# In[ ]:


from ResNet import ResNet34
res = ResNet34(NUM_CLASSES)


# In[ ]:


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_accuracy', value=0.995, verbose=0):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
       
        if current > self.value:
            print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


# In[ ]:


res.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=[tf.keras.metrics.CategoricalAccuracy(), 'accuracy'])
res.fit(train_x, train_y, epochs=20,
          callbacks=[tensorboard_callback,EarlyStopping()],
          validation_data=(val_x, val_y))


# In[ ]:


res.evaluate(val_x, val_y)


# In[ ]:


res.save("resnet_model")


# In[ ]:


X_test = []
for dirs in os.listdir("./Data/Test"):
    X_test.append(cv2.imread("./Data/Test/"+ dirs))
X_test = np.array(X_test)


# In[ ]:


y_prediction = model_res.predict(X_test)
y_pred = []
for i in range(len(X_test)):
    y_pred.append(np.argmax(ypred[i]))
sub = [] 
i = 0
for dirs in os.listdir("./Data/Test"):
    sub.append((dirs, classes[y_pred[i]]))
    i+=1


# In[ ]:


subs = pd.DataFrame(sub)
subs = subs.rename(columns = {1:"Category", 0:"Id"})
subs.set_index('Id', inplace=True)
subs.to_csv("subs.csv")

