import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

class Vgg16:
    def __init__(self):
        self.first_fc_unit = 4096
        # self._feat_stride = [16, ]
        # self._feat_compress = [1. / float(self._feat_stride[0]), ]
        # self._scope = 'vgg_16'
        # self._input_shape = input_shape
        # self._batch_size = batch_size
        # self._class_num = class_num

    def image_to_head(self, inputs):
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(inputs)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='SAME')(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='SAME')(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='SAME')(x)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='SAME')(x)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer='l2')(x)
        return x

    def head_to_tail(self, inputs):
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(units=4096, activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(units=4096, activation='relu', kernel_regularizer='l2')(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        return x

    def build_graph(self, input_shape, class_num):
        inputs = tf.keras.Input(shape=input_shape)
        x = self.image_to_head(inputs=inputs)
        # x = self.head_to_tail(inputs=x)
        x = tf.keras.layers.Dense(units=class_num)(x)
        outputs = tf.keras.models.Model(inputs=inputs, outputs=x)
        return outputs


# class Vgg16_keras:
#     def __init__(self):
#         pass
#
#     def build_graph(self, class_num):
#         base_model = VGG16(weights='imagenet', include_top=False)
#         x = base_model.output
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)
#         x = tf.keras.layers.Dense(1024, activation='relu')(x)
#         predictions = tf.keras.layers.Dense(class_num, activation='softmax')(x)
#
#         # for layer in base_model.layers:
#         #     layer.trainable = False
#
#         model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
#         return model

if __name__ == "__main__":
    vgg16 = Vgg16()
    vgg16_model = vgg16.build_graph((500, 500, 3),10)
    vgg16_model.summary(line_length=100)

    # vgg16.first_fc_unit = 100
    # vgg16_model.summary(line_length=100)