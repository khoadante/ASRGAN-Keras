import tensorflow as tf
from keras import Model
from keras.layers import SeparableConv2D, LeakyReLU, concatenate


class ResidualDenseBlock(Model):
  def __init__(self):
    super(ResidualDenseBlock, self).__init__()
    self.conv_1 = SeparableConv2D(32, (3, 3), strides=(1, 1), padding="SAME")
    self.conv_2 = SeparableConv2D(32, (3, 3), strides=(1, 1), padding="SAME")
    self.conv_3 = SeparableConv2D(32, (3, 3), strides=(1, 1), padding="SAME")
    self.conv_4 = SeparableConv2D(32, (3, 3), strides=(1, 1), padding="SAME")
    self.conv_5 = SeparableConv2D(64, (3, 3), strides=(1, 1), padding="SAME")

    self.activation_1 = LeakyReLU(alpha=0.2)
    self.activation_2 = LeakyReLU(alpha=0.2)
    self.activation_3 = LeakyReLU(alpha=0.2)
    self.activation_4 = LeakyReLU(alpha=0.2)
  
  def call(self, x):
    x1 = self.activation_1(self.conv_1(x))
    x2 = self.activation_2(self.conv_2(concatenate([x, x1])))
    x3 = self.activation_3(self.conv_3(concatenate([x, x1, x2])))
    x4 = self.activation_4(self.conv_4(concatenate([x, x1, x2, x3])))
    x5 = self.conv_5(concatenate([x, x1, x2, x3, x4]))

    # Emperically, we use 0.2 to scale the residual for better performance
    return x5 * 0.2 + x

class ARRDB(tf.keras.Model):
  def __init__(self):
    super(ARRDB, self).__init__()
    self.res_1 = ResidualDenseBlock()
    self.res_2 = ResidualDenseBlock()
    self.res_3 = ResidualDenseBlock()
  
  def call(self, x_input):
    x = self.res_1(x_input)
    x = self.res_2(x)
    x = self.res_3(x)

    return x * 0.2 + x_input

