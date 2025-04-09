import tensorflow as tf
from tensorflow.keras import layers, models
#Model definition for ResEMoteNet
#Squeeze-and-Excitation Block for channel-wise attention
class SEBlock(tf.keras.layers.Layer):

    #Initialize the SEBlock with input channels and reduction factor
    def __init__(self, in_channels, reduction=16, **kwargs):

        super(SEBlock, self).__init__(**kwargs)  # Call parent constructor
        self.in_channels = in_channels  # Store the number of input channels
        self.reduction = reduction  # Store the reduction factor
        self.avg_pool = layers.GlobalAveragePooling2D()  # Global average pooling layer
        self.fc1 = layers.Dense(in_channels // reduction, activation='relu', use_bias=False)  # First dense layer
        self.fc2 = layers.Dense(in_channels, activation='sigmoid', use_bias=False)  # Second dense layer

    #Forward pass for the SEBlock
    def call(self, x):

        y = self.avg_pool(x)  # Apply global average pooling to input
        y = self.fc1(y)  # Pass through the first dense layer
        y = self.fc2(y)  # Pass through the second dense layer
        y = tf.reshape(y, [-1, 1, 1, self.in_channels])  # Reshape the output to match input dimensions
        return x * y  # Scale input by the output of the SE block

    #Get the configuration of the SEBlock for serialization
    def get_config(self):

        config = super(SEBlock, self).get_config()  # Get parent configuration
        config.update({
            'in_channels': self.in_channels,  # Add in_channels to config
            'reduction': self.reduction  # Add reduction to config
        })
        return config  # Return the updated configuration

#Residual Block for building deep networks with skip connections
class ResidualBlock(tf.keras.layers.Layer):

    #Initialize the ResidualBlock with input and output channels
    def __init__(self, in_ch, out_ch, stride=1, **kwargs):

        super(ResidualBlock, self).__init__(**kwargs)  # Call parent constructor
        self.in_ch = in_ch  # Store the number of input channels
        self.out_ch = out_ch  # Store the number of output channels
        self.stride = stride  # Store the stride value

        # Define the first convolutional layer
        self.conv1 = layers.Conv2D(out_ch, kernel_size=3, strides=stride, padding="same")
        self.bn1 = layers.BatchNormalization()  # Batch normalization after conv1
        # Define the second convolutional layer
        self.conv2 = layers.Conv2D(out_ch, kernel_size=3, padding="same")
        self.bn2 = layers.BatchNormalization()  # Batch normalization after conv2

        # Define shortcut connection
        if stride != 1 or in_ch != out_ch:
            self.shortcut = models.Sequential([
                layers.Conv2D(out_ch, kernel_size=1, strides=stride),  # 1x1 conv for matching dimensions
                layers.BatchNormalization()  # Batch normalization for shortcut
            ])
        else:
            self.shortcut = lambda x: x  # Identity shortcut if dimensions match

    #Forward pass for the ResidualBlock
    def call(self, x):

        out = tf.nn.relu(self.bn1(self.conv1(x)))  # Apply conv1, batch norm, and ReLU
        out = self.bn2(self.conv2(out))  # Apply conv2 and batch norm
        out += self.shortcut(x)  # Add shortcut connection
        return tf.nn.relu(out)  # Apply ReLU activation to output

    #Get the configuration of the ResidualBlock for serialization
    def get_config(self):

        config = super(ResidualBlock, self).get_config()  # Get parent configuration
        config.update({
            'in_ch': self.in_ch,  # Add in_ch to config
            'out_ch': self.out_ch,  # Add out_ch to config
            'stride': self.stride  # Add stride to config
        })
        return config  # Return the updated configuration

#ResEmoteNet model for emotion recognition
class ResEmoteNet(tf.keras.Model):

    #Initialize the ResEmoteNet with the number of output classes
    def __init__(self, num_classes=7, **kwargs):

        super(ResEmoteNet, self).__init__(**kwargs)  # Call parent constructor
        self.conv1 = layers.Conv2D(64, kernel_size=3, padding="same")  # First convolutional layer
        self.bn1 = layers.BatchNormalization()  # Batch normalization after conv1
        self.conv2 = layers.Conv2D(128, kernel_size=3, padding="same")  # Second convolutional layer
        self.bn2 = layers.BatchNormalization()  # Batch normalization after conv2
        self.conv3 = layers.Conv2D(256, kernel_size=3, padding="same")  # Third convolutional layer
        self.bn3 = layers.BatchNormalization()  # Batch normalization after conv3
        self.se = SEBlock(256)  # Squeeze-and-Excitation block

        # Define residual blocks
        self.res_block1 = ResidualBlock(256, 512, stride=2)  # First residual block
        self.res_block2 = ResidualBlock(512, 1024, stride=2)  # Second residual block
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)  # Third residual block

        self.pool = layers.GlobalAveragePooling2D()  # Global average pooling layer
        self.fc1 = layers.Dense(1024, activation="relu")  # First fully connected layer
        self.fc2 = layers.Dense(512, activation="relu")  # Second fully connected layer
        self.fc3 = layers.Dense(256, activation="relu")  # Third fully connected layer
        self.dropout = layers.Dropout(0.5)  # Dropout layer for regularization
        self.fc4 = layers.Dense(num_classes)  # Output layer for classification

    #Forward pass for the ResEmoteNet
    def call(self, x):

        x = tf.nn.relu(self.bn1(self.conv1(x)))  # Apply conv1, batch norm, and ReLU
        x = layers.MaxPooling2D()(x)  # Max pooling after conv1
        x = tf.nn.relu(self.bn2(self.conv2(x)))  # Apply conv2, batch norm, and ReLU
        x = layers.MaxPooling2D()(x)  # Max pooling after conv2
        x = tf.nn.relu(self.bn3(self.conv3(x)))  # Apply conv3, batch norm, and ReLU
        x = layers.MaxPooling2D()(x)  # Max pooling after conv3
        x = self.se(x)  # Apply Squeeze-and-Excitation block

        x = self.res_block1(x)  # Pass through first residual block
        x = self.res_block2(x)  # Pass through second residual block
        x = self.res_block3(x)  # Pass through third residual block

        x = self.pool(x)  # Global average pooling
        x = self.fc1(x)  # Pass through first fully connected layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Pass through second fully connected layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Pass through third fully connected layer
        x = self.dropout(x)  # Apply dropout
        return self.fc4(x)  # Output layer for classification

    #Get the configuration of the ResEmoteNet for serialization
    def get_config(self):

        config = super(ResEmoteNet, self).get_config()  # Get parent configuration
        config.update({
            'num_classes': 7  # Add num_classes to config
        })
        return config  # Return the updated configuration

    #Create an instance of ResEmoteNet from a configuration dictionary
    @classmethod
    def from_config(cls, config):

        return cls(**config)  # Instantiate ResEmoteNet with the provided configuration
