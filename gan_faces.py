from keras.models import Sequential
from keras.layers import Conv2D, Dropout, LeakyReLU, Flatten, Conv2DTranspose
from keras.layers import Dense, Reshape
from keras.optimizers import Adam


import numpy as np
from matplotlib import pyplot

def load_data():
  data = np.load('full_numpy_bitmap_face.npy')
  #data = data.swapaxes(0,1)
  data = data.reshape((data.shape[0],28,28))
  data = np.expand_dims(data, axis=-1)
  data = data / 255
  return data

def initialize_discriminator(layer_shape=(28, 28, 1)):
  model = Sequential()
  model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', input_shape=layer_shape))
  model.add(LeakyReLU())
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  opt = Adam(lr = 0.0001, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

def initialize_generator(latent_dim = 100):
  model = Sequential()
  #generate 7X7 img 
  model.add(Dense(128*7*7, input_dim=latent_dim))
  model.add(LeakyReLU())
  model.add(Reshape((7,7,128)))
  #upscale to 14X14 img
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU())
  #upscale to 28X28 img
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU())
  model.add(Conv2D(1, (3,3), activation='sigmoid', padding='same'))
  return model

def initialize_GAN(d_model, g_model):
  d_model.trainable = False
  model = Sequential()
  model.add(g_model)
  model.add(d_model)
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)
  return model

def generate_latent_space(latent_dim,  num_sample):
  #generate latent space from standard distribution
  x_input = np.random.randn(latent_dim * num_sample).reshape(num_sample, latent_dim)
  return x_input

def generate_real_data(num_sample):
  x_data = load_data()
  x_index = np.random.choice(x_data.shape[0], num_sample, replace=False)
  x_data = x_data[x_index]
  y = np.ones((num_sample, 1))
  return x_data, y

def generate_fake_data(model, latent_dim, num_sample):
  x = generate_latent_space(latent_dim, num_sample)
  x = model.predict(x)
  y = np.zeros((num_sample, 1))
  return x, y

def summarize_performance(iter, g_model, d_model, latent_dim, num_sample):
  x_real, y_real = generate_real_data(num_sample)
  _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
  x_fake, y_fake = generate_fake_data(g_model, latent_dim, num_sample)
  _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
  print('Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
  save_plot(x_fake, iter)
  filename = 'generator_model_%d.h5' % (iter)
  g_model.save(filename)

def save_plot(examples, iter, n=5):
  for i in range(n**2):
    pyplot.subplot(n,n,1+i)
    pyplot.axis('off')
    pyplot.imshow(examples[i,:,:,0], cmap='gray_r')
  filename = 'generated_plot_img_%d' % (iter)
  pyplot.savefig(filename)
  pyplot.close()

def train(d_model, g_model, gan_model, latent_dim=100, iter=10000, num_sample=1000, verbose=False):
  y_gan = np.ones((num_sample,1))
  for i in range(iter):
    #generate data for generator, discriminator, and GAN
    x_fake, y_fake = generate_fake_data(g_model, latent_dim, num_sample)
    x_real, y_real = generate_real_data(num_sample)
    x_gan = generate_latent_space(latent_dim, num_sample)
    x, y = np.vstack((x_fake, x_real)), np.vstack((y_fake, y_real))
    #train
    #train discriminator
    d_loss, _ = d_model.train_on_batch(x, y)
    #train generator via composite model
    g_loss = gan_model.train_on_batch(x_gan, y_gan)
    if verbose:
      print('iter: %d | d_loss: %0.3f | g_loss: %0.3f' % (i, d_loss, g_loss))
    if i % 100 == 0:
      summarize_performance(i, g_model, d_model, latent_dim, num_sample)




  
#def initialize_discriminator():

#data = np.load('full_numpy_bitmap_face.npy')
#print(data.shape)
#data = data.reshape((161666,28,28))
#print(data.shape)
#print(data)
#pyplot.imshow(data[3,:,:], cmap='gray_r')
if __name__=='__main__':
  data = load_data()
  print(data.shape)
  d_model = initialize_discriminator()
  g_model = initialize_generator()
  gan_model = initialize_GAN(d_model, g_model)
  train(d_model, g_model, gan_model)
  