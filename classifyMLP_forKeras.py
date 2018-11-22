# -*- coding: utf-8 -*-

import argparse
from sklearn.datasets import load_svmlight_files

# TensorFlow from Keras
from keras.models import model_from_json

# 設定したサンプル数ごとに勾配を更新
b_size = 256

def main():

    # コマンドライン引数の読み込み
    args = readArgs()

    # モデルの読み込み
    model = model_from_json(open(args.model + '.json', 'r').read())

    # データの読み込み
    test, testTag = load_svmlight_files([args.test], n_features=model.input_shape[-1])
    test = test.toarray()

    # モデルに学習したパラメータを読み込む
    model.load_weights(args.model + '.hdf5')

    kekka = model.predict_classes(test, batch_size=b_size, verbose=1)
    # kekka = model.predict_proba(test, batch_size=b_size, verbose=1)

    fp = open(args.output, 'w')

    for i in kekka:
        fp.write("{}\n".format(str(i[0])))

    fp.close()


# コマンドライン引数の処理関数
# 引数: なし
# 返値: 処理した引数の辞書
def readArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('test', type=str, help='テストデータ')
    parser.add_argument('model', type=str, default='model', help='モデルデータ')
    parser.add_argument('output', type=str, default='output', help='結果出力ファイル')

    return parser.parse_args()


if __name__ == '__main__':
    main()
