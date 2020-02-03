# nmist_GAN
Won Jun Lee

[Art & Ai](https://www.vip.gatech.edu/teams/art-ai) team in Vertically Integrated Projects at Georgia Institute of Technology.   

This project is to understand how GAN works with nmist dataset.

It is recommended to run in [Google Colab](https://colab.research.google.com).

## discriminator.py

Includes discriminator model.

To test discriminator.py

```
python discriminator.py
```

Running discriminator.py will create/train discriminator model for MNIST dataset and display summary of the model.

## generator.py

Includes generator model. 

To test generator.py

```
python generator.py
```

Running generator.py will create generator model for MNIST dataset and display fake images created by generator model.

## Attribution
This implementation uses some code from Jason Brownlee's [How to Develop a GAN for Generating MNIST Handwritten Digits](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/)
