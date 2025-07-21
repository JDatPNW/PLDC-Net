from tensorflow.keras import layers, Model
from keras.src import backend

dense_block_growth_rate = 32
dense_kernel_size = 3
dense_reduction = 0.5

image_size = (224,224,3)
num_classes = 80

def DensePreAmble(x, postfix):
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1 # bn_axis is the batch_norm axis
    prefix = "DensePreAmble_" + postfix + "_"


    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name=prefix+"ZeroPadding2D_1")(x)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name=prefix+"Conv2D_1")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=prefix+"BatchNormalization")(x) # x = layers.LayerNormalization()(x) from ConvNeXt?
    x = layers.Activation("swish", name=prefix + "Swish_1")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=prefix+"ZeroPadding2D_2")(x)
    x = layers.MaxPooling2D(3, strides=2, name=prefix+"MaxPooling2D")(x)
    return x

def DenseBlock(x, growth_rate, kernel_size, num_layers, postfix):
    prefix = "DenseBlock_" + postfix + "_"
    for i in range(num_layers):
        prefix_layer = prefix + str(i) + "_"
        dense_connection = x # stores the input so it can be concactinated
        
        bn_axis = 3 if backend.image_data_format() == "channels_last" else 1 # bn_axis is the batch_norm axis
        
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=prefix_layer+"BatchNormalization")(x)
        x = layers.Activation("swish", name=prefix_layer + "Swish_1")(x)

        x = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=prefix_layer+"Conv2D_Bottleneck")(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=prefix_layer+"BatchNormalization_2")(x) 

        x = layers.Activation("swish", name=prefix_layer + "Swish_2")(x)

        x = layers.Conv2D(growth_rate, kernel_size, padding="same", use_bias=False, name=prefix_layer+"Conv2D_2")(x) 

        x = layers.Concatenate(axis=bn_axis, name=prefix_layer+"Concatenate")([x, dense_connection]) 
    return x

def TransitionBlock(x, reduction, postfix):
    prefix = "TransitionBlock_" + postfix
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=prefix+"BatchNormalization")(x)
    x = layers.Activation("swish", name=prefix + "Swish_1")(x)
    x = layers.Conv2D(int(x.shape[bn_axis] * reduction), 1, use_bias=False, name=prefix+"Conv2D")(x)
    x = layers.AveragePooling2D(2, strides=2, name=prefix+"AveragePooling2D")(x)
    return x

def DensePostamble(x):
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = layers.Activation("swish", name="Swish_postamble")(x)
    return x


def ChannelAttentionBlock(inputs, reduction_ratio=8):    
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)
    
    fc1 = layers.Dense(inputs.shape[-1] // reduction_ratio, activation='relu', use_bias=False)
    fc2 = layers.Dense(inputs.shape[-1], use_bias=False)
    
    avg_out = fc2(fc1(avg_pool))
    max_out = fc2(fc1(max_pool))
    
    channel_attention_output = avg_out + max_out
    channel_attention_output = layers.Activation('sigmoid')(channel_attention_output)
    
    return layers.Multiply()([inputs, channel_attention_output])


input = layers.Input(image_size)

preamble = DensePreAmble(input, "1")

model = DenseBlock(preamble, growth_rate=dense_block_growth_rate, kernel_size=dense_kernel_size, num_layers=6, postfix="2")
model = TransitionBlock(model, dense_reduction, "3")

model = DenseBlock(model, growth_rate=dense_block_growth_rate, kernel_size=dense_kernel_size, num_layers=12, postfix="4")
model = TransitionBlock(model, dense_reduction, "5")

model = DenseBlock(model, growth_rate=dense_block_growth_rate, kernel_size=dense_kernel_size, num_layers=48, postfix="6")
model = TransitionBlock(model, dense_reduction, "7")

model = DenseBlock(model, growth_rate=dense_block_growth_rate, kernel_size=dense_kernel_size, num_layers=32, postfix="8")
model = DensePostamble(model)

model = ChannelAttentionBlock(model)

model = layers.GlobalAveragePooling2D()(model)
model = layers.Dense(num_classes, activation="softmax")(model)

pldc_net = Model(inputs=input, outputs=model)

pldc_net.summary()
