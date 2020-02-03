#GaTech Art&AI VIP
#Won Jun Lee
#Source: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import Conv2D, LeakyReLU, Conv2DTranspose
from keras.optimizers import Adam
#from keras import backend as K
from PIL import Image
import numpy as np
from matplotlib import pyplot

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_row, img_col = x_train[0].shape

def initialize_generator(latent_space=100):
  model = Sequential()
  
  #generating 128, 7X7 images
  model.add(Dense(128*7*7, input_dim=latent_space))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Reshape((7,7,128)))

  #upsample images to 14X14
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  #upsample images to 28X28
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(1, (3,3), activation='sigmoid', padding='same'))
  return model


def generate_latent_space(latent_space,  num_sample):
  #generating random points in latent space
  #using standard normal distribution
  x_input = np.random.randn(latent_space * num_sample).reshape(num_sample, latent_space)
  return x_input

def generate_fake_samples(model, latent_space, num_sample):
  #generate fake samples for generator
  x = generate_latent_space(latent_space, num_sample)
  #debug
  #print(x.shape)
  #debug
  x = model.predict(x)
  y = np.zeros((num_sample, 1))
  return x,y


if __name__=='__main__':
  model = initialize_generator()
  model.summary()
  latent_space = 100
  n_samples = 25
  x, _ = generate_fake_samples(model, latent_space, n_samples)
  for i in range(n_samples):
    pyplot.subplot(5,5,1+i)
    pyplot.axis('off')
    pyplot.imshow(x[i,:,:,0], cmap='gray_r')
  pyplot.show()

