import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model

#Initialize the CNN model parameters.
class CNNModel:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):

        self.input_shape = input_shape  # Set the input shape
        self.num_classes = num_classes  # Set the number of classes

    #A convolutional block consisting of Conv2D, BatchNormalization, Activation, MaxPooling2D, and Dropout.
    def _conv_block(self, x, filters, kernel_size, pool_size, dropout_rate):
        for filter_size in filters:  # Iterate through the filter sizes
            x = Conv2D(filter_size, kernel_size, padding='same', activation='relu')(x)  # Apply Conv2D
            x = BatchNormalization()(x)  # Apply Batch Normalization
        x = MaxPooling2D(pool_size=pool_size)(x)  # Apply MaxPooling
        x = Dropout(dropout_rate)(x)  # Apply Dropout for regularization
        return x  # Return the output tensor

    #Build the CNN model architecture.
    def build_model(self):

        inputs = Input(shape=self.input_shape)  # Define the input layer

        # Convolutional Blocks
        x = self._conv_block(inputs, filters=[64, 64], kernel_size=(3, 3), pool_size=(2, 2), dropout_rate=0.25)  # First conv block
        x = self._conv_block(x, filters=[128, 128], kernel_size=(3, 3), pool_size=(2, 2), dropout_rate=0.35)  # Second conv block
        x = self._conv_block(x, filters=[256, 256], kernel_size=(3, 3), pool_size=(2, 2), dropout_rate=0.45)  # Third conv block

        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)  # Apply Global Average Pooling

        # Fully Connected Layers
        x = Dense(128, activation='relu')(x)  # First Dense Layer
        x = BatchNormalization()(x)  # Apply Batch Normalization
        x = Dropout(0.5)(x)  # Apply Dropout

        # Output Layer
        outputs = Dense(self.num_classes, activation='softmax')(x)  # Output layer with softmax activation

        model = Model(inputs=inputs, outputs=outputs)  # Create the Keras Model
        return model  # Return the model