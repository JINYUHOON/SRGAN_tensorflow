import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19

class Modules:
    def __init__(self, input_shape = (None,None,3)):
        self.input_shape = input_shape
    
    def vgg_extractor(self):
        vgg = VGG19(include_top=False, weights = 'imagenet', input_shape= self.input_shape)
        return keras.Model(vgg.input, vgg.layers[20].output)
    
    def generator(self):
        inputs = layers.Input(self.input_shape)
        out = layers.Conv2D(64, 9, 1, padding='same')(inputs)

        out = res = layers.PReLU(shared_axes=[1,2])(out)
        
        for _ in range(5):
            out = self.residual_block(out)

        out = layers.Conv2D(64, 3, 1, padding='same')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Add()([res, out])
        
        for _ in range(2):
            out = self.upsample_block(out)
        out = layers.Conv2D(3, 9, 1, padding='same', activation='sigmoid')(out)

        return keras.Model(inputs, out)
    
    def disriminator(self):
        inputs = layers.Input(self.input_shape)
        
        out = layers.Conv2D(64, 3, 1, padding='same')(inputs)
        out = layers.LeakyReLU()(out)

        out = layers.Conv2D(64, 3, 2, padding='same')(out)
        out = layers.BatchNormalization()(out)
        out = layers.LeakyReLU()(out)

        for filter in [128, 256, 512]:
            out = self.residual_disc(out, filter)
        
        out = layers.Dense(1024)(out)
        out = layers.LeakyReLU()(out)
        out = layers.Dense(1, activation='sigmoid')(out)
        
        return keras.Model(inputs, out)
        
    def residual_block(self, x):
        out = layers.Conv2D(64, 3, 1, padding='same')(x)
        out = layers.BatchNormalization()(out)
        out = layers.PReLU(shared_axes=[1,2])(out)
        out = layers.Conv2D(64, 3, 1, padding='same')(out)
        out = layers.BatchNormalization()(out)
        return layers.Add()([x, out])
    
    def upsample_block(self, x):
        out = layers.Conv2D(256, 3, 1, padding='same')(x)
        out = layers.Lambda(lambda x : tf.nn.depth_to_space(x, 2))(out)
        return layers.PReLU(shared_axes=[1,2])(out)
    
    
    def residual_disc(self, x, n_filters):
        out = layers.Conv2D(n_filters, 3, 1, padding='same')(x)
        out = layers.BatchNormalization()(out)
        out = layers.LeakyReLU()(out)
        out = layers.Conv2D(n_filters, 3, 2, padding='same')(out)
        out = layers.BatchNormalization()(out)
        out = layers.LeakyReLU()(out)
        return out
    
        
