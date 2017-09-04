import numpy as np
import tensorflow as tf


def GeneratorCNN(z,batch_size, hidden_num, output_num,repeat_num,data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as scope:
        if reuse:
            scope.reuse_variables()
        num_output = int(np.prod([8, 8, hidden_num]))
        x = linear(z, num_output, "g_fc1")
        x = reshape(batch_size, x, 8, 8, hidden_num, data_format)
        for idx in range(repeat_num):
            gstr1 = "g_%d" % ((2 * idx) + 1)
            gstr2 = "g_%d" % ((2 * idx) + 2)
            x = tf.nn.elu(conv2d(x,hidden_num,name = gstr1,data_format=data_format))
            x = tf.nn.elu(conv2d(x,hidden_num,name = gstr2,data_format=data_format))
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)
        gstr3 = "g_%d" % ((2 * idx) + 3)
        out = conv2d(x,3,name = gstr3)
    return out


def DiscriminatorCNN(x, batch_size, input_channel, z_num, cont_num, cat_num, repeat_num, hidden_num, data_format,reuse = False):
    with tf.variable_scope("D", reuse=reuse) as scope:
        if reuse:
            scope.reuse_variables()
        # Encoder
        x = tf.nn.elu(conv2d(x,hidden_num,name = "e_0"))
        #prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            estr1 = "e_%d" % ((3 * idx) + 1)
            estr2 = "e_%d" % ((3 * idx) + 2)
            estr3 = "e_%d" % ((3 * idx) + 3)
            x = tf.nn.elu(conv2d(x,channel_num,name = estr1))
            x = tf.nn.elu(conv2d(x,channel_num,name = estr2))
            if idx < repeat_num - 1:
                x = tf.nn.elu(conv2d(x, channel_num,d_h=2,d_w=2, name=estr3))
        x1 = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = linear(x1, z_num, "e_fc1")
        qx2 = linear(x1, 256, 'q_h1_lin')
        qx3 = linear(qx2, 128, 'q_h2_lin')
        q_out_cat = tf.nn.softmax(linear(qx2, cat_num, 'q_cat_out'))
        q_out_cont = tf.nn.tanh(linear(qx3, cont_num, 'q_conts_out_lin'))

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        x = linear(x, num_output, "d_fc1")
        x = reshape(2*batch_size, x, 8, 8, hidden_num, data_format)
        for idx in range(repeat_num):
            dstr1 = "d_%d" % ((2 * idx) + 1)
            dstr2 = "d_%d" % ((2 * idx) + 2)
            x = tf.nn.elu(conv2d(x,hidden_num,name = dstr1))
            x = tf.nn.elu(conv2d(x,hidden_num,name = dstr2))
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)
        dstr3 = "d_%d" % ((2 * idx) + 3)
        out = conv2d(x,input_channel,name = dstr3)
    return out, z, q_out_cont, q_out_cat


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # print(shape)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(batch_size,x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [batch_size, c, h, w])
    else:
        x = tf.reshape(x, [batch_size, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)

def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
       name="conv2d", data_format = None):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME',data_format=data_format)
    return conv


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias