from tensorflow import keras
from keras.layers import Conv2D, Flatten, GlobalAveragePooling2D, MaxPooling2D, Input, Dense
from keras.models import Model

INPUT_SIZE= (224, 224, 3)

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

def make_model(architecture, input_shape, num_of_classes=2):

    inputs = Input(input_shape)
    x = None

    for c, layer in enumerate(architecture):
        if(c==0):
            x = Conv2D(filters=layer[0], kernel_size=layer[1], strides=layer[2], padding=layer[3], activation="relu")(inputs)
            continue
            
        if(type(layer) == tuple):
            x = Conv2D(filters=layer[0], kernel_size=layer[1], strides=layer[2], padding=layer[3], activation="relu")(x)

        if(layer == "MP"):
            x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        
    label = Dense(num_of_classes, activation="sigmoid", name="label")(x)
    
    model = Model([inputs], [label])
    return model


if(__name__ == "__main__"):
    model = make_model(DARKNET19_ARCHITECTURE, INPUT_SIZE)
    model.build(INPUT_SIZE)
    model.summary()

