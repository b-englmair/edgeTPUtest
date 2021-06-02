
import tensorflow as tf
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import numpy as np
import pickle


def main():
    # Load and reshape data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = tf.keras.utils.to_categorical(y_train)

    # Load evaluation templates from pickle
    x_style = pickle.load(open('mnist_test_style.pickle', 'rb'))['x_style']

    # Determine dimensions
    img_width, img_height = x_train.shape[1], x_train.shape[2]
    num_channels = 1  # since all images are in gray scalepy
    input_size = [img_width, img_height, num_channels]

    # Build encoder model
    input_img = tf.keras.Input(shape=input_size, name='input_image')
    # convolution and pooling layers
    layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(input_img)
    layer = tf.keras.layers.MaxPooling2D((2, 2))(layer)
    layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(layer)
    # layer = tf.keras.layers.MaxPooling2D((2, 2))(layer)
    # layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(layer)
    conv_shape = backend.int_shape(layer)
    layer = tf.keras.layers.Flatten()(layer)
    # dense layers
    encoder_output = tf.keras.layers.Dense(128, activation="relu")(layer)
    # build model and summarize it
    encoder = tf.keras.Model(input_img, encoder_output, name='Encoder')
    print(encoder.summary())

    # Build decoder model
    decoder_input = tf.keras.Input(shape=128, name='decoder_input')
    # dense layers
    layer = tf.keras.layers.Dense(128, activation="relu")(decoder_input)
    layer = tf.keras.layers.Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(layer)
    # deconvolution and upscale layers
    layer = tf.keras.layers.Reshape([conv_shape[1], conv_shape[2], conv_shape[3]])(layer)
    layer = tf.keras.layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(layer)
    layer = tf.keras.layers.UpSampling2D((2, 2))(layer)
    layer = tf.keras.layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(layer)
    # layer = tf.keras.layers.UpSampling2D((2, 2))(layer)
    # layer = tf.keras.layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(layer)
    decoder_output = tf.keras.layers.Conv2DTranspose(num_channels, 3, padding='same', activation='sigmoid')(layer)
    # build model and summarize it
    decoder = tf.keras.Model(decoder_input, decoder_output, name='Decoder')
    print(decoder.summary())
    # build reconstructed image
    reconstructed_img = decoder(encoder_output)

    # Build autoencoder
    autoencoder = tf.keras.Model(input_img, reconstructed_img, name='VAE')
    print(autoencoder.summary())

    # Compile autoencoder
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss="mean_squared_error")

    # Train autoencoder
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1)
    history = autoencoder.fit(x_train, x_train, epochs=1, batch_size=128, shuffle=True, validation_split=0.1,
                      callbacks=[early_stop])

    # Generate novel images
    fig, axs = plt.subplots(12, 10)
    plt.gray()
    for row, img in enumerate(x_style):
        imgs = np.empty([10, img.shape[0], img.shape[1], 1])
        for col in range(10):
            img = np.reshape(img, newshape=(1, 28, 28, 1))
            imgs[col] = img
        reconstructed_imgs = autoencoder.predict(imgs)
        for col in range(10):
            reconstructed_img = reconstructed_imgs[col]
            reconstructed_img = reconstructed_img.reshape(28, 28)
            axs[row, col].imshow(reconstructed_img)
            axs[row, col].get_xaxis().set_visible(False)
            axs[row, col].get_yaxis().set_visible(False)
    plt.savefig(f'output.png', dpi=600)


if __name__ == '__main__':
    main()


