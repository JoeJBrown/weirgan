from __future__ import print_function

import os
#import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

from models import *
from utils import save_image

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')
        self.q_lr = tf.Variable(config.q_lr, name='q_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')
        self.q_lr_update = tf.assign(self.q_lr, tf.maximum(self.q_lr * 0.5, config.lr_lower_boundary), name='q_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, height, width, self.channel = get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.q_step = config.q_step

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False

            self.build_test_model()

    def train(self):
        z_fixed_2 = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
        z_int = np.random.randint(0, self.cat_dim, [self.batch_size]).astype(np.int32)
        z_fixed = np.zeros((self.batch_size, self.cat_dim))
        z_fixed[np.arange(self.batch_size), z_int] = 1
        z_fixed = np.reshape(z_fixed, [self.batch_size, self.cat_dim])
        z_fixed_1 = np.random.uniform(-1, 1, [self.batch_size, self.cont_num]).astype(np.float32)
        z_fixed = np.concatenate([z_fixed, z_fixed_1, z_fixed_2], axis=1)

        x_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_update": self.k_update,
                "measure": self.measure,
            }
            fetch_dict_q = {
                "k_update": self.k_update_q,
                "measure": self.measure,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "q_loss": self.q_loss,
                    "k_t": self.k_t,
                })
                fetch_dict_q.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "q_loss": self.q_loss,
                    "k_t": self.k_t,
                })

            if step % self.q_step == 0:
                result = self.sess.run(fetch_dict_q)
            else:
                result = self.sess.run(fetch_dict)
            measure = result['measure']
            measure_history.append(measure)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
                      format(step, self.max_step, d_loss, g_loss, measure, k_t))

            if step % (self.log_step * 10) == 0:
                x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update,self.q_lr_update])

    def build_model(self):

        self.x = self.data_loader
        x = norm_img(self.x)
        self.k_t = tf.Variable(0., trainable=False, name='k_t')
        self.cat_num = 1
        self.cont_num = 10

        self.z_noise = tf.random_uniform((tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.cat_dim = self.cat_num
        self.cat_list = [self.cat_num]
        self.latent_cat_in = tf.constant(np.random.randint(0, self.cat_dim, [self.batch_size, 1]).astype(np.int32),dtype=tf.int32)
        self.z_lats = tf.one_hot(indices=self.latent_cat_in, depth=self.cat_dim)
        self.z_lats = tf.reshape(self.z_lats, [-1, self.cat_dim])

        self.latent_cont_in = tf.random_uniform((tf.shape(x)[0], self.cont_num), minval=-1.0, maxval=1.0)

        self.z_lat = tf.concat([self.z_lats, self.latent_cont_in, self.z_noise], axis=1)
        # self.z_lat = tf.concat([self.latent_cont_in, self.z_noise], axis=1)

        G = GeneratorCNN(self.z_lat,self.batch_size, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=False)

        d_out, self.D_z, q_out_cont, q_out_cat = DiscriminatorCNN(
                tf.concat([G, x], 0),self.batch_size, self.channel, self.z_num,self.cont_num,self.cat_num, self.repeat_num,
                self.conv_hidden_num,self.data_format)

        AE_G, AE_x = tf.split(d_out, 2)
        # we only want the q from our generated points
        self.q_out_cat, _ = tf.split(q_out_cat,2)
        self.q_out_cont, _ = tf.split(q_out_cont,2)
        self.G = denorm_img(G, self.data_format)
        self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        g_optimizer, d_optimizer, q_optimizer = optimizer(self.g_lr), optimizer(self.d_lr),optimizer(self.q_lr)

        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - G))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.abs(AE_G - G))

        self.q_cat_loss = -tf.reduce_mean(self.z_lats * tf.log(self.q_out_cat + 1e-10), reduction_indices=1)
        self.q_cont_loss = tf.reduce_mean(0.5 * tf.square(self.latent_cont_in - self.q_out_cont)+ 1e-10)

        self.q_cont_loss = tf.reduce_mean(self.q_cont_loss)
        self.q_cat_loss = tf.reduce_mean(self.q_cat_loss)

        self.q_loss = 0.5*tf.add(self.q_cat_loss, self.q_cont_loss)
        #####
        t_vars = tf.trainable_variables()
        self.e_vars = [var for var in t_vars if 'e_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.q_vars = [var for var in t_vars if 'q_' in var.name]

        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.e_vars+self.d_vars)
        g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.g_vars)
        q_optim = q_optimizer.minimize(self.q_loss, var_list = self.q_vars+self.g_vars)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)


        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        with tf.control_dependencies([d_optim, g_optim, q_optim]):
            self.k_update_q = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))


        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.image("AE_G", self.AE_G),
            tf.summary.image("AE_x", self.AE_x),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])

    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num+self.cat_num+self.cont_num], tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z_lat)

        G_z_r = GeneratorCNN(self.z_r,self.batch_size,self.conv_hidden_num, self.channel,
            self.repeat_num, self.data_format, reuse = True)

        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.G, {self.z_lat: inputs})

        if path is None and save:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        items = {
            'real': inputs,
            'fake': x_fake,
        }
        for key, img in items.items():
            if img is None:
                continue
            # if img.shape[3] in [1, 3]:
            #     img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x = self.sess.run(self.AE_x, {self.x: img})
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_x, {self.D_z: z})

    def test(self):
        root_path = "./test_images/"
        all_G_z = None
        for step in range(10):
            z_int = np.random.randint(0, self.cat_dim, [self.batch_size]).astype(np.int32)
            z_fixed = np.zeros((self.batch_size, self.cat_dim))
            z_fixed[np.arange(self.batch_size), z_int] = 1
            z_fixed = np.reshape(z_fixed, [self.batch_size, self.cat_dim])
            z_fixed_1 = np.random.uniform(-1, 1, [self.batch_size, self.cont_num]).astype(np.float32)

            z_fixed_2 = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            z_fixed = np.concatenate([z_fixed, z_fixed_1, z_fixed_2], axis=1)

            G_z = self.generate(z_fixed, path=os.path.join(root_path, "test{}_G_z.png".format(step)))
            if all_G_z is None:
                all_G_z = G_z
            else:
                all_G_z = np.concatenate([all_G_z, G_z])
            save_image(G_z, '{}/G_z{}.png'.format(root_path, step))
        #np.save("{}/all_G_z".format(root_path), all_G_z)
        save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)

    def real_ims(self):
        root_path = "./test_images/"
        real_im_list = None
        for step in range(10):
            real_im = self.get_image_from_loader()
            if real_im_list is None:
                real_im_list = real_im
            else:
                real_im_list = np.concatenate([real_im_list, real_im])
        print(real_im_list.shape)
        np.save("{}/real_ims".format(root_path), real_im_list)


    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x
