from discriminator import initialize_discriminator
from generator import initialize_generator, generate_latent_space, 
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

def initialize_GAN (d_model, g_model):
  d_model.trainable = False
  model = Sequential()
  model.add(g_model)
  model.add(d_model)
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)
  return model

def train_GAN(gan_model, latent_space=100, iter=100, num_batch=256, verbose=False):
  if verbose:
    print("start training GAN")
  for i in range(iter):
    x = generate_latent_space(latent_space, num_batch)
    y = np.ones((num_batch,1))
    gan_model.train_on_batch(x,y)
    if verbose:
      print("finished iter: {}".format(i))

#def train_full(d_model, g_model, gan_model, latent_space=100, iter=100, num_batch=256, verbose=False):


#d_model.summary()
#g_model.summary()

if __name__ == '__main__':
  d_model = initialize_discriminator(layer_shape=(28,28,1))
  g_model = initialize_generator(100)
  model = initialize_GAN(d_model, g_model)
  train_GAN(model, verbose=True)