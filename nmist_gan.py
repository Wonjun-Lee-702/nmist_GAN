from discriminator import initialize_discriminator, generate_real_data, load_training_data
from generator import initialize_generator, generate_latent_space, generate_fake_samples
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot

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

    #we want discriminator to think that the samples from generator are real
    #this is because when we train discriminator, it will think that
    #any other samples with < 1 outputs is fake. 
    y = np.ones((num_batch,1))
    gan_model.train_on_batch(x,y)
    if verbose:
      print("finished iter: {}".format(i))

def train_full(d_model, g_model, gan_model, latent_space=100, iter=100, num_batch=256, verbose=False):
  y_gan = np.ones((num_batch,1))
  for i in range(iter):
    x_real, y_real = generate_real_data(num_batch)
    x_fake, y_fake = generate_fake_samples(g_model, latent_space, num_batch)
    #print(x_real.shape)
    #print(x_fake.shape)
    x, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
    x_gan = generate_latent_space(latent_space, num_batch)
    
    #first, update the discriminator model with real and fake samples
    d_loss, _ = d_model.train_on_batch(x, y)

    #second, update the generator via the composite model
    g_loss = gan_model.train_on_batch(x_gan, y_gan)
    if verbose:
      print("iter: %d, discriminator loss: %.3f, generator loss: %.3f"% (i+1, d_loss, g_loss))
    if i % 100 == 9:
      summarize_performance(i, g_model, d_model, latent_space)

def summarize_performance(iter, g_model, d_model, latent_dim, n_sample=100):
  x_real, y_real = generate_real_data(n_sample)
  _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
  x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_sample)
  _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
  print('Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
  save_plot(x_fake, iter)
  filename = 'generator_model_%03d.h5' % (iter + 1)
  g_model.save(filename)

def save_plot(examples, iter, n=10):
  for i in range(n**2):
    pyplot.subplot(n,n,1+i)
    pyplot.axis('off')
    pyplot.imshow(examples[i,:,:,0], cmap='gray_r')
  filename = "generated_plot_e%03d.png" % (iter+ 1 )
  pyplot.savefig(filename)
  pyplot.close()

#d_model.summary()
#g_model.summary()

if __name__ == '__main__':
  d_model = initialize_discriminator(layer_shape=(28,28,1))
  g_model = initialize_generator(100)
  model = initialize_GAN(d_model, g_model)
  #train_GAN(model, verbose=True)
  train_full(d_model, g_model, model, iter=10000, verbose=True)
  