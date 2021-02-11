from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Conv2D


def build_model(input_size):
    """
    モデルを構築
    # 引数
        input_size : List, 入力画像のサイズ
    # 戻り値
        model : Keras model
    """
    input_shape = (input_size[0], input_size[1], 3)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(3, kernel_size=(3, 3), padding='same'))
    model.add(Activation('sigmoid'))
    return model