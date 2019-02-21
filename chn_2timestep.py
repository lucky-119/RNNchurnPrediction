from random import random
import numpy as np
import pandas as pd
import math

from keras.models import Sequential  
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
import matplotlib.pylab as plt
from keras import metrics

df = pd.read_csv('churnbig.csv', sep=",",header=None)


length=len(df)
X_train_length=int(math.ceil(0.7*length))

X_train=df.iloc[:X_train_length,1:]
y_train=df.iloc[:X_train_length,0]

X_test_length=length-X_train_length
X_test=df.iloc[:X_test_length,1:]
y_test=df.iloc[:X_test_length,0]


print X_train.head()
print y_train.head()
print X_test.head()
print y_test.head()

X_train=X_train.as_matrix()
y_train=y_train.as_matrix()

X_train = X_train.reshape(len(X_train)/2, 2, 15)
y_train = y_train.reshape(len(y_train)/2, 2)


X_test=X_test.as_matrix()
y_test=y_test.as_matrix()

X_test = X_test.reshape(len(X_test)/2, 2, 15)
y_test = y_test.reshape(len(y_test)/2, 2)

in_out_neurons = 15 
hidden_neurons = 20

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,input_shape=(2,15)))
#model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dense(2, input_dim=hidden_neurons))  
model.add(Activation("linear"))  
model.compile(loss="binary_crossentropy", optimizer="rmsprop",  metrics=[metrics.binary_accuracy])
model.summary()

model2 = Sequential()  
model2.add(LSTM(hidden_neurons, return_sequences=True,input_shape=(2,15)))  
model2.add(LSTM(20, return_sequences=True))  
model2.add(Dropout(0.2))  
model2.add(LSTM(20, return_sequences=False))  
model2.add(Dropout(0.2))  
model2.add(Dense(2, input_dim=hidden_neurons))  
model2.add(Activation("sigmoid"))  
model2.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=[metrics.binary_accuracy])  




history = model.fit(X_train, y_train, batch_size=28, epochs=50)
history2 = model2.fit(X_train, y_train, batch_size=28, epochs=50)

predicted = model.predict(X_test)
predicted
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))


print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
    
plt.plot(history2.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# "Loss"
plt.plot(history2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
    
plt.rcParams["figure.figsize"] = (13, 9)
plt.plot(predicted[:100][:,0],"--")
#plt.plot(predicted[:100][:],"--")
plt.plot(y_train[:100][:,0],":")
#plt.plot(y_train[:100][:,1],":")
plt.legend(["Prediction 0", "Prediction 1", "Test 0", "Test 1"])
plt.show()
