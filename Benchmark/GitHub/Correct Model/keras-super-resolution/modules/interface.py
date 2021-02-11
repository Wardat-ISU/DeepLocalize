import numpy as np
from PIL import Image


def show(image):
    """
    画像データを表示
    # 引数
        image : Numpy array, 画像データ
    """
    image = image[0] * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.show()


def get_input():
    """
    標準入力から文字列を取得
    # 戻り値
        value : String, 入力値
    """
    value = input('>> ')
    value = value.rstrip()
    if value == 'q':
        exit()
    return value