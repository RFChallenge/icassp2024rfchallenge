import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def get_unet_model(input_shape, k_sz=3, long_k_sz=101, lr=0.0003, k_neurons=32):
    n_window = input_shape[0]
    n_ch = 2

    in0 = layers.Input(shape=input_shape)
    x = in0
    
    x = layers.BatchNormalization()(x)
    
    upsamp_blocks = []
    
    for n_layer, k in enumerate([8, 8, 8, 8, 8]):
        if n_layer == 0:
            conv = layers.Conv1D(k_neurons * k, long_k_sz, activation="relu", padding="same")(x)
        else:
            conv = layers.Conv1D(k_neurons * k, k_sz, activation="relu", padding="same")(x)
        
        conv = layers.Conv1D(k_neurons * k, k_sz, activation="relu", padding="same")(conv)        
        pool = layers.MaxPooling1D(2)(conv)
        if n_layer == 0:
            pool = layers.Dropout(0.25)(pool)
        else:
            pool = layers.Dropout(0.5)(pool)
        
        upsamp_blocks.append(conv)
        x = pool
    
    # Middle
    convm = layers.Conv1D(k_neurons * 8, k_sz, activation="relu", padding="same")(x)
    convm = layers.Conv1D(k_neurons * 8, k_sz, activation="relu", padding="same")(convm)
    
    x = convm
    for n_layer, k in enumerate([8, 8, 4, 2, 1]):
        deconv = layers.Conv1DTranspose(k_neurons * k, k_sz, strides=2, padding="same")(x)
        uconv = layers.concatenate([deconv, upsamp_blocks[-(n_layer+1)]])
        uconv = layers.Dropout(0.5)(uconv)
        uconv = layers.Conv1D(k_neurons * k, k_sz, activation="relu", padding="same")(uconv)
        uconv = layers.Conv1D(k_neurons * k, k_sz, activation="relu", padding="same")(uconv)
        
        x = uconv

    output_layer = layers.Conv1D(n_ch, 1, padding="same", activation=None)(x)
    
    x_out = output_layer
    supreg_net = Model(in0, x_out, name='supreg')
    
    supreg_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.losses.MeanSquaredError()])    

    return supreg_net
