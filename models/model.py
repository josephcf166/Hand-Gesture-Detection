from tensorflow import keras
from keras.layers import Conv2D, Flatten, GlobalAveragePooling2D, MaxPooling2D, Input, Dense, LeakyReLU, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.models import Model

INPUT_SIZE= (448, 448, 3)

DARKNET19_ARCHITECTURE = [
    # (number of filters, size, stride, padding)
    (32, 3, 1, "same"),
    "MP",
    (64, 3, 1, "same"),
    "MP",
    (128, 3, 1, "same"),
    (64, 1, 1, "same"),
    (128, 3, 1, "same"),
    "MP",
    (256, 3, 1, "same"),
    (128, 1, 1, "same"),
    (256, 3, 1, "same"),
    "MP",
    (512, 3, 1, "same"),
    (256, 1, 1, "same"),
    (512, 3, 1, "same"),
    (256, 1, 1, "same"),
    (512, 3, 1, "same"),
    "MP",
    (1024, 3, 1, "same"),
    (512, 1, 1, "same"),
    (1024, 3, 1, "same"),
    (512, 1, 1, "same"),
    (1024, 3, 1, "same")
]

def build_model(architecture, input_shape, num_of_classes=2):

    inputs = Input(input_shape)
    x = None

    for c, layer in enumerate(architecture):
        if(c==0):
            x = Conv2D(filters=layer[0], kernel_size=layer[1], strides=layer[2], padding=layer[3], use_bias=False, kernel_regularizer=l2(0.01))(inputs)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization()(x)
            continue
            
        if(type(layer) == tuple):
            x = Conv2D(filters=layer[0], kernel_size=layer[1], strides=layer[2], padding=layer[3], use_bias=False, kernel_regularizer=l2(0.01))(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization()(x)

        if(layer == "MP"):
            x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    bbox = Dense(4, activation="sigmoid", name="bbox")(x)
    # label = Dense(1, activation="sigmoid", name="label")(x)
    
    model = Model([inputs], [bbox])
    return model


if(__name__ == "__main__"):
    model = build_model(DARKNET19_ARCHITECTURE, INPUT_SIZE)
    model.build(INPUT_SIZE)
    model.summary()

