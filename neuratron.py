from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Input
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.models import load_model
from keras.regularizers import L2
import keras, os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class NeuratronLayer(keras.layers.Layer):
    def __init__(self, filters, kernel_size, num_sub_layers, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.num_sub_layers = num_sub_layers
        self.strides = strides

    def build(self, input_shape):
        self.sub_layers = []
        for _ in range(self.num_sub_layers):
            self.sub_layers.append(Sequential([
                BatchNormalization(),
                Conv2D(self.filters, self.kernel_size, 
                      strides=self.strides, 
                      activation=self.activation,
                      padding='same',
                      kernel_regularizer=L2(1e-6)),
                MaxPooling2D(2)
            ]))
        super().build(input_shape)

    def call(self, inputs):
        # Verifica se a entrada é 4D
        if len(inputs.shape) != 4:
            raise ValueError(f"Input deve ser 4D. Recebido: {inputs.shape}")
            
        outputs = []
        for sub_layer in self.sub_layers:
            outputs.append(sub_layer(inputs))
        
        # Corrigido: removido o "+inputs" que causava problemas
        stacked = tf.stack(outputs, axis=-1)
        return tf.reduce_mean(stacked, axis=-1)

    def compute_output_shape(self, input_shape):
        # Calcula shape após conv+pool
        batch_size = input_shape[0]
        rows = input_shape[1] // 2  # Redução do MaxPooling2D
        cols = input_shape[2] // 2
        return (batch_size, rows, cols, self.filters)



if __name__ == '__main__':

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    encoder = keras.layers.CategoryEncoding(num_tokens=np.max(train_labels)+1, output_mode="one_hot")
    print("Shape antes do BatchNorm:", train_images.shape)
    bn_out = BatchNormalization()(train_images)
    print("Shape após BatchNorm:", bn_out.shape)
    
    train_labels = encoder(train_labels)
    test_labels = encoder(test_labels)
    model = None
    if os.path.exists('neuratron.keras'):
        model = load_model('neuratron.keras')

    else:
        model = Sequential([
            # Camada de entrada precisa especificar o shape completo (incluindo canais)
            Input(shape=(28, 28, 1)),  # Adicionado explicitamente
            
            # Removido 'units' e 'input_shape' dos parâmetros - não são necessários
            NeuratronLayer(filters=64, kernel_size=3, strides=1, num_sub_layers=3),
            NeuratronLayer(filters=128, kernel_size=3, strides=1, num_sub_layers=3),
            NeuratronLayer(filters=128, kernel_size=3, num_sub_layers=3),
            
            # Achatar para Dense
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(10, activation='softmax')
        ])

    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.FalseNegatives(),
    ])

    model.fit(np.expand_dims(train_images, axis=-1), train_labels)
    predictions = model.predict(np.expand_dims(test_images, axis=-1))
    plt.figure(figsize=(10, 10))

    for i in range(9):  # Mostrar 9 exemplos
        plt.subplot(3, 3, i+1)
        plt.imshow(test_images[i].squeeze(), cmap='gray')  # Remove dimensão de canal se necessário
        plt.title(f"Real: {test_labels[i]}\nPrevisto: {np.argmax(predictions[i])}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    model.save('neuratron.keras')
