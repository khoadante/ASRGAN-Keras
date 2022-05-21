from keras import Model, Input
from keras.layers import Conv2D, LeakyReLU, UpSampling2D, ReLU, MaxPooling2D
from keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow_addons.layers import SpectralNormalization
from networks.blocks import RRDB


class RRDBNet(Model):
  def __init__(self):
    super(RRDBNet, self).__init__()
    self.first_conv = Conv2D(64, 3, strides=(1, 1), padding="SAME") # padding 1
    self.rb_1 = RRDB()
    self.rb_2 = RRDB()
    self.rb_3 = RRDB()
    self.rb_4 = RRDB()
    self.rb_5 = RRDB()
    self.rb_6 = RRDB()
    self.conv_body = Conv2D(64, 3, strides=(1, 1), padding="SAME") # padding 1

    self.up_1 = UpSampling2D()
    self.conv_1 = Conv2D(64, 3, strides=(1, 1), padding="SAME") # padding 1
    self.activation_1 = LeakyReLU(alpha=0.2)

    self.up_2 = UpSampling2D()
    self.conv_2 = Conv2D(64, 3, strides=(1, 1), padding="SAME") # padding 1
    self.activation_2 = LeakyReLU(alpha=0.2)

    self.conv_3 = Conv2D(64, 3, strides=(1, 1), padding="SAME") # padding 1
    self.activation_3 = LeakyReLU(alpha=0.2)
    self.conv_4 = Conv2D(3, 3, strides=(1, 1), padding="SAME") # padding 1
    
  def call(self, x_input):
    x = self.first_conv(x_input)
    x1 = self.rb_1(x)
    x1 = self.rb_2(x1)
    x1 = self.rb_3(x1)
    x1 = self.rb_4(x1)
    x1 = self.rb_5(x1)
    x1 = self.rb_6(x1)
    x1 = self.conv_body(x1)

    x = x + x1

    x = self.up_1(x)
    x = self.conv_1(x)
    x = self.activation_1(x)

    x = self.up_2(x)
    x = self.conv_2(x)
    x = self.activation_2(x)

    x = self.conv_3(x)
    x = self.activation_3(x)
    x = self.conv_4(x)

    return x

class UNetDiscriminator(Model):
  def __init__(self):
    super(UNetDiscriminator, self).__init__()

    self.conv_0 = SpectralNormalization(Conv2D(64, 3, strides=(1, 1), padding="SAME", use_bias=False))
    self.conv_1 = SpectralNormalization(Conv2D(128, 4, strides=(2, 2), padding="SAME", use_bias=False))
    self.conv_2 = SpectralNormalization(Conv2D(256, 4, strides=(2, 2), padding="SAME", use_bias=False))
    self.conv_3 = SpectralNormalization(Conv2D(512, 4, strides=(2, 2), padding="SAME", use_bias=False))

    self.conv_4 = SpectralNormalization(Conv2D(256, 3, strides=(1, 1), padding="SAME", use_bias=False))
    self.conv_5 = SpectralNormalization(Conv2D(128, 3, strides=(1, 1), padding="SAME", use_bias=False))
    self.conv_6 = SpectralNormalization(Conv2D(64, 3, strides=(1, 1), padding="SAME", use_bias=False))

    self.conv_7 = SpectralNormalization(Conv2D(64, 3, strides=(1, 1), padding="SAME", use_bias=False))
    self.conv_8 = SpectralNormalization(Conv2D(64, 3, strides=(1, 1), padding="SAME", use_bias=False))

    self.conv_9 = SpectralNormalization(Conv2D(1, 3, strides=(1, 1), padding="SAME", use_bias=False))

  def call(self, x):
    x0 = LeakyReLU(alpha=0.2)(self.conv_0(x)) # 256x256
    x1 = LeakyReLU(alpha=0.2)(self.conv_1(x0)) # 128x128
    x2 = LeakyReLU(alpha=0.2)(self.conv_2(x1))  # 64x64
    x3 = LeakyReLU(alpha=0.2)(self.conv_3(x2)) # 32x32

    x3 = UpSampling2D(interpolation="bilinear")(x3)
    x4 = LeakyReLU(alpha=0.2)(self.conv_4(x3)) # 64x64

    x4 = x4 + x2

    x4 = UpSampling2D(interpolation="bilinear")(x4)
    x5 = LeakyReLU(alpha=0.2)(self.conv_5(x4)) # 128x128

    x5 = x5 + x1

    x5 = UpSampling2D(interpolation="bilinear")(x5)
    x6 = LeakyReLU(alpha=0.2)(self.conv_6(x5)) # 256x256

    x6 = x6 + x0

    out = LeakyReLU(alpha=0.2)(self.conv_7(x6)) # 256x256
    out = LeakyReLU(alpha=0.2)(self.conv_8(out)) # 256x256
    out = self.conv_9(out)

    return out

def create_vgg19_custom_model():
  # Block 1
  input = Input(shape=(None, None, 3))
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(input)
  x = ReLU()(x)
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
  x = ReLU()(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
  x = ReLU()(x)
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
  x = ReLU()(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
  x = ReLU()(x)
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
  x = ReLU()(x)
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
  x = ReLU()(x)
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv4')(x)
  x = ReLU()(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
  x = ReLU()(x)
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
  x = ReLU()(x)
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
  x = ReLU()(x)
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv4')(x)
  x = ReLU()(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
  x = ReLU()(x)
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
  x = ReLU()(x)
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
  x = ReLU()(x)
  x = Conv2D(512, (3, 3), padding='same', name='block5_conv4')(x)
  x = ReLU()(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  return Model(input, x, name="vgg19_features")

class Vgg19FeaturesModel(Model):
  def __init__(self):
    super(Vgg19FeaturesModel, self).__init__()
    original_vgg = VGG19(include_top=False, weights='imagenet')
    self.vgg_model = create_vgg19_custom_model()
    self.vgg_model.set_weights(original_vgg.get_weights())

    layers = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4", "block5_conv4"]
    outputs = [self.vgg_model.get_layer(name).output for name in layers]

    self.model = Model([self.vgg_model.input], outputs)
    self.model.trainable = False
      
  def call(self, x):
    x = preprocess_input(x * 255.0)
    return self.model(x)

