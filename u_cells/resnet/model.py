import tensorflow.keras.models as keras_model
import tensorflow.keras.layers as keras_layer


class BatchNorm(keras_layer.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    """ The identity_block is the block that has no conv layer at shortcut

    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers

    Returns:

    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras_layer.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                           use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = keras_layer.Activation('relu')(x)

    x = keras_layer.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                           name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = keras_layer.Activation('relu')(x)

    x = keras_layer.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                           use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = keras_layer.Add()([x, input_tensor])
    x = keras_layer.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """ Conv_block is the block that has a conv layer at shortcut

    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers

    Note:
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well

    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras_layer.Conv2D(nb_filter1, (1, 1), strides=strides,
                           name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = keras_layer.Activation('relu')(x)

    x = keras_layer.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                           name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = keras_layer.Activation('relu')(x)

    x = keras_layer.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                                    '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = keras_layer.Conv2D(nb_filter3, (1, 1), strides=strides,
                                  name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = keras_layer.Add()([x, shortcut])
    x = keras_layer.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]

    # Stage 1
    x = keras_layer.ZeroPadding2D((3, 3))(input_image)
    x = keras_layer.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = keras_layer.Activation('relu')(x)
    C1 = x = keras_layer.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x

    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None

    return [C1, C2, C3, C4, C5]


def to_rpn(pyramid_size, C2, C3, C4, C5):
    """

    Args:
        pyramid_size:
        C2:
        C3:
        C4:
        C5:

    Returns:

    """
    P5 = keras_layer.Conv2D(pyramid_size, (1, 1), name='fpn_c5p5')(C5)
    P4 = keras_layer.Add(name="fpn_p4add")([
        keras_layer.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        keras_layer.Conv2D(pyramid_size, (1, 1), name='fpn_c4p4')(C4)])
    P3 = keras_layer.Add(name="fpn_p3add")([
        keras_layer.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        keras_layer.Conv2D(pyramid_size, (1, 1), name='fpn_c3p3')(C3)])
    P2 = keras_layer.Add(name="fpn_p2add")([
        keras_layer.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        keras_layer.Conv2D(pyramid_size, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = keras_layer.Conv2D(pyramid_size, (3, 3), padding="same", name="fpn_p2")(P2)
    P3 = keras_layer.Conv2D(pyramid_size, (3, 3), padding="same", name="fpn_p3")(P3)
    P4 = keras_layer.Conv2D(pyramid_size, (3, 3), padding="same", name="fpn_p4")(P4)
    P5 = keras_layer.Conv2D(pyramid_size, (3, 3), padding="same", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = keras_layer.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    return [P2, P3, P4, P5, P6]
