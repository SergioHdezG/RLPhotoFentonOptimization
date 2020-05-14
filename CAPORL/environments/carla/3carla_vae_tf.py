from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
from tensorflow.python.keras.models import model_from_json
input_shape_glob = 128

class vae_class:
    def __init__(self):
        self.build_model()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    def plot_results(self, models,
                     data,
                     batch_size=128,
                     model_name="vae_mnist"):
        """Plots labels and MNIST digits as function of 2-dim latent vector

        # Arguments
            models (tuple): encoder and decoder models
            data (tuple): test data and label
            batch_size (int): prediction batch size
            model_name (string): which model is using this function
        """

        encoder, decoder = models
        x_test, y_test = data
        os.makedirs(model_name, exist_ok=True)

        filename = os.path.join(model_name, "vae_mean.png")
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = encoder.predict(x_test,
                                       batch_size=batch_size)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig(filename)
        plt.show()

        filename = os.path.join(model_name, "digits_over_latent.png")
        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        plt.show()


    def generate_and_save_images(self, latent_dim, test_data):
        figsize = 5
        # num_examples_to_generate = figsize*figsize
        # encoder, decoder = models

        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        # random_vector_for_generation = np.random.normal(size=(num_examples_to_generate, latent_dim))

        # predictions = decoder.predict(random_vector_for_generation)
        # fig1 = plt.figure(figsize=(figsize, figsize))
        #
        # for i in range(num_examples_to_generate):
        #     plt.subplot(figsize, figsize, i+1)
        #     if predictions.shape[3] == 1:
        #         plt.imshow(predictions[i, :, :, 0], cmap='gray')
        #     else:
        #         plt.imshow(predictions[i, :, :, :])
        # plt.axis('off')
        # plt.savefig('figura_6.png')



        examples_index = np.random.choice(test_data.shape[0], figsize*2)
        examples = test_data[examples_index]
        # z_mean, z_log_var, z = encoder.predict(examples)
        # images = decoder.predict(z)

        images_rgb, images_seg, images_dep = self.predict(examples)

        fig2 = plt.figure()
        for i in range(figsize*2):
            plt.subplot(2, figsize*2, i+1)
            if examples.shape[3] == 1:
                plt.imshow(examples[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(examples[i])

            plt.subplot(2, figsize*2, i + 1 + figsize*2)
            if examples.shape[3] == 1:
                plt.imshow(images_rgb[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(images_rgb[i])
        plt.axis('off')

        fig3 = plt.figure()
        for i in range(figsize*2):
            plt.subplot(2, figsize*2, i+1)
            if examples.shape[3] == 1:
                plt.imshow(examples[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(examples[i])

            plt.subplot(2, figsize*2, i + 1 + figsize*2)
            if examples.shape[3] == 1:
                plt.imshow(images_seg[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(images_seg[i])
        plt.axis('off')

        fig4 = plt.figure()
        for i in range(figsize*2):
            plt.subplot(2, figsize*2, i+1)
            if examples.shape[3] == 1:
                plt.imshow(examples[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(examples[i])

            plt.subplot(2, figsize*2, i + 1 + figsize*2)
            if examples.shape[3] == 1:
                plt.imshow(images_dep[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(images_dep[i])
        plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        # plt.savefig('figura_7.png')
        plt.show()



    # MNIST dataset
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    kernel_size = 3
    latent_dim = 128

    def build_model(self, input_shape=(128, 128, 3)):
        self.rgb_channel = tf.placeholder(dtype=tf.float32, shape=(None, *input_shape), name="rgb_channel_in")
        self.seg_channel = tf.placeholder(dtype=tf.float32, shape=(None, *input_shape), name="seg_channel_in")
        self.dep_channel = tf.placeholder(dtype=tf.float32, shape=(None, *input_shape), name="dep_channel_in")


        # VAE model = encoder + decoder
        encoder, latent_shape = self.build_encoder(input_shape)
        encoder.summary()



        decoder_rgb = self.build_decoder(input_shape, latent_shape, 'rgb')
        decoder_seg = self.build_decoder(input_shape, latent_shape, 'seg')
        decoder_dep = self.build_decoder(input_shape, latent_shape, 'dep')

        decoder_rgb.summary()
        # plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

        # instantiate VAE model
        z_mean, z_log_var, z = encoder(self.rgb_channel)
        outputs_rgb = decoder_rgb(z)
        outputs_seg = decoder_seg(z)
        outputs_dep = decoder_dep(z)

        beta = 1
        with tf.variable_scope('loss'):
            recons_loss_rgb = self.reconstruction_loss(input_shape, self.rgb_channel, outputs_rgb)
            recons_loss_seg = self.reconstruction_loss(input_shape, self.seg_channel, outputs_seg)
            recons_loss_dep = self.reconstruction_loss(input_shape, self.dep_channel, outputs_dep)

            kl_loss = beta * self.kl_loss(z_log_var,  z_mean)

            vae_loss = K.mean(recons_loss_rgb + recons_loss_seg + recons_loss_dep + kl_loss)

            self.loss = vae_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.loss)

        # self.prueba = [recons_loss_rgb, recons_loss_seg, kl_loss, vae_loss]

        self.predict_images = decoder_rgb(z_mean), decoder_seg(z_mean), decoder_dep(z_mean)

        return encoder, decoder_rgb, None, None # decoder_seg, decoder_dep

    def reconstruction_loss(self, input_shape, inputs, outputs):
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))
        reconstruction_loss *= input_shape[0] * input_shape[1]
        return reconstruction_loss

    def kl_loss(self, z_log_var, z_mean):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss

    def build_encoder(self, input_shape):
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')

        x = inputs
        x = Conv2D(filters=16,
                   kernel_size=self.kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
        x = Conv2D(filters=16,
                   kernel_size=self.kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
        x = Conv2D(filters=16,
                   kernel_size=self.kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
        x = Conv2D(filters=16,
                   kernel_size=self.kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
        # shape info needed to build decoder model
        shape = K.int_shape(x)
        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder, shape


    def build_decoder(self, input_shape, shape, type=''):
        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling_'+type)
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu', name='dense_'+type)(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        x = Conv2DTranspose(filters=16,
                            kernel_size=self.kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        x = Conv2DTranspose(filters=16,
                            kernel_size=self.kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        x = Conv2DTranspose(filters=16,
                            kernel_size=self.kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        x = Conv2DTranspose(filters=16,
                            kernel_size=self.kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        outputs = Conv2DTranspose(filters=input_shape[2],
                                  kernel_size=self.kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output_'+type)(x)
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder_'+type)
        return decoder

    def predict(self, input):
        images_rgb, images_seg, images_dep = self.sess.run(self.predict_images, feed_dict={self.rgb_channel: input})
        return images_rgb, images_seg, images_dep

    def fit(self, x_train_rgb, x_train_seg, x_train_dep, batch_size=128, epochs=10, shuffle=False, validation_data=None):
        x_test_rgb = []
        if validation_data is not None:
            x_test_rgb = validation_data[0]
            x_test_seg = validation_data[1]
            x_test_dep = validation_data[2]

        if shuffle:
            shuffle_train_index = np.array(random.sample(range(len(x_train_rgb)), len(x_train_rgb)))
            x_train_rgb = np.array(x_train_rgb)[shuffle_train_index]
            x_train_seg = np.array(x_train_seg)[shuffle_train_index]
            x_train_dep = np.array(x_train_dep)[shuffle_train_index]

        n_test_samples = len(x_test_rgb)
        n_train_samples = len(x_train_rgb)

        import gc
        gc.collect()

        print("train samples: ", len(x_train_rgb), " val_samples: ", n_test_samples)
        for epoch in range(epochs):
            mean_loss = []
            for batch in range(n_train_samples//batch_size + 1):
                i = batch * batch_size
                j = (batch+1) * batch_size

                if j >= n_train_samples:
                    j = n_train_samples

                train_batch_rgb = x_train_rgb[i:j]
                train_batch_seg = x_train_seg[i:j]
                train_batch_dep = x_train_dep[i:j]
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.rgb_channel: train_batch_rgb,
                                                                               self.seg_channel: train_batch_seg,
                                                                               self.dep_channel: train_batch_dep
                                                                               })
                mean_loss.append(loss)

            val_loss = 0
            if validation_data is not None:
                val_loss = self.sess.run(self.loss, feed_dict={self.rgb_channel: x_test_rgb,
                                                               self.seg_channel: x_test_seg,
                                                               self.dep_channel: x_test_dep
                                                               })
            mean_loss = np.mean(mean_loss)
            print('epoch', epoch, "\tloss: ", mean_loss, "\tval_loss: ", val_loss)

    def test(self, x_test, dir, name):
        self.load(dir, name)
        self.generate_and_save_images(self.latent_dim, x_test)



    def train(self, x_train, x_test, epochs, batch_size):
        x_train_rgb = x_train[0]
        x_train_seg = x_train[1]
        x_train_dep = x_train[2]

        x_test_rgb = x_test[0]
        x_test_seg = x_test[1]
        x_test_dep = x_test[2]

        # train the autoencoder
        self.fit(x_train_rgb, x_train_seg, x_train_dep,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=[x_test_rgb, x_test_seg, x_test_dep])

        self.generate_and_save_images(self.latent_dim, x_test_rgb)
        self.save('/home/serch/TFM/IRL2/CAPORL/environments/carla/autoencoder/vae_tf')

    def save(self, name):
        self.saver.save(self.sess, name)
        print("Saved model to disk")

    def load(self, dir, name):
        name = os.path.join(dir, name)
        loaded_model = tf.train.import_meta_graph(name + '.meta')
        loaded_model.restore(self.sess, tf.train.latest_checkpoint(dir))

def main():
    print("Loading dataset..")
    images_rgb = np.load('/home/serch/TFM/IRL2/CAPORL/environments/carla/rgb_carla.npy')
    images_rgb = np.array(images_rgb[-3000:])
    images_seg = np.load('/home/serch/TFM/IRL2/CAPORL/environments/carla/seg_carla.npy')
    images_seg = np.array(images_seg[-3000:])
    images_dep = np.load('/home/serch/TFM/IRL2/CAPORL/environments/carla/dep_carla.npy')
    images_dep = np.array(images_dep[-3000:])


    test_index = np.random.choice(images_rgb.shape[0], int(images_rgb.shape[0] * 0.20), replace=False)
    x_test_rgb = images_rgb[test_index]
    x_test_seg = images_seg[test_index]
    x_test_dep = images_dep[test_index]

    # test_index = np.array(sorted(test_index, reverse=False))
    train_index_mask = np.isin(range(images_rgb.shape[0]), test_index)
    x_train_rgb = images_rgb[~train_index_mask]  # Delete test samples from train data
    x_train_seg = images_seg[~train_index_mask]  # Delete test samples from train data
    x_train_dep = images_dep[~train_index_mask]  # Delete test samples from train data

    x_train_rgb = x_train_rgb.astype('float32') / 255
    x_train_seg = x_train_seg.astype('float32') / 255
    x_train_dep = x_train_dep.astype('float32') / 255

    x_test_rgb = x_test_rgb.astype('float32') / 255
    x_test_seg = x_test_seg.astype('float32') / 255
    x_test_dep = x_test_dep.astype('float32') / 255



    import gc
    gc.collect()

    # network parameters
    batch_size = 128
    epochs = 20


    vae = vae_class()
    # vae.test(x_test_rgb, '/home/serch/TFM/IRL2/CAPORL/environments/carla/autoencoder', 'vae_tf')
    vae.train([x_train_rgb, x_train_seg, x_train_dep], [x_test_rgb, x_test_seg, x_test_dep], epochs, batch_size)

main()