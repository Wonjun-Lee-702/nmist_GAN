#GaTech Art&AI VIP
#Won Jun Lee
#Source: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ReLU, LeakyReLU
from keras.optimizers import Adam
#from keras import backend as K
from PIL import Image
import numpy as np
from matplotlib import pyplot

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_row, img_col = x_train[0].shape

#print(len(x_train))
#print(len(y_train))
#data = x_train[0]
#img = Image.fromarray(data)
#img.save('my.png')
#pyplot.imshow(img, cmap='gray_r')


def initialize_discriminator (layer_shape = (img_row, img_col, 1)):
  #discriminator model will be sequential
  model = Sequential()

  #adding Con2D layer
  #more information can be found: https://keras.io/layers/convolutional/
  #filters -> # of outputs
  #kernel_size -> basically, a size of lense(filter) for our CNN
  #strides -> #of pixels we want to shift out lense(filter) https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148
  #padding -> https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', input_shape=layer_shape))
  #Rectified Linear Activation Function
  #used to change output to >0.0(positive) or 0(negative)
  #model.add(ReLU())
  
  #used LeakyRelU instead. 
  #read https://www.quora.com/What-is-the-dying-ReLU-problem-in-neural-networks
  #more avtivation function can be found here: https://keras.io/activations/
  #https://towardsdatascience.com/deep-study-of-a-not-very-deep-neural-network-part-2-activation-functions-fd9bd8d406fc
  model.add(LeakyReLU(alpha=0.2))

  #Dropout is a technique used to prevent a model from overfitting. 
  #Dropout works by randomly setting the outgoing edges of hidden units 
  #(neurons that make up hidden layers) to 0 at each update of the training phase.
  #https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(1,activation='sigmoid'))
  optimizer = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

def load_training_data():

  #so that our img is 3D 
  x = np.expand_dims(x_train, axis=-1)

  #normalize pixel values
  x = x / 255
  return x

def generate_real_data(num_sample):
  index = np.random.choice(x_train.shape[0], num_sample, replace=False)
  x = load_training_data()
  x = x[index]
  y = np.ones((num_sample,1))
  return x,y

def generate_fake_data(num_sample):
  x = np.random.ranf(num_sample*img_row*img_col)
  x = x.reshape(num_sample, img_row, img_col, 1)
  y = np.zeros((num_sample, 1))
  return x,y

def train_discriminator(model, iter=100, num_sample = 256, verbose=False):
  for i in range(iter):
    x_real, y_real = generate_real_data(num_sample)
    x_fake, y_fake = generate_fake_data(num_sample)
    _, real_acc = model.train_on_batch(x_real, y_real)
    _, fake_acc = model.train_on_batch(x_fake, y_fake)
    if verbose:
      print("iter: {}, real: {}, fake: {}".format(i,real_acc*100, fake_acc*100))



if __name__ == "__main__":
  print("image shape: ", img_row, img_col)
  print("data shape: ")
  print("x_train: ", x_train.shape)
  print("y_train: ", y_train.shape)
  print("x_test: ", x_test.shape)
  print("y_test: ", y_test.shape)
  discriminator_model = initialize_discriminator()
  print("***discriminator_model summary***")
  discriminator_model.summary()
  x_train = load_training_data()
  train_discriminator(discriminator_model, verbose=True)