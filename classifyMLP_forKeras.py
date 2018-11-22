# -*- coding: utf-8 -*-

from argparse import ArgumentParser

from sklearn.datasets import load_svmlight_files
from keras.models import load_model

# 設定したサンプル数ごとに勾配を更新
b_size = 256


def main():
    model = load_model(args.model, compile=False)

    test, testTag = load_svmlight_files([args.svm_test_file], n_features=model.input_shape[-1])
    test = test.toarray()

    kekka = model.predict_classes(test, batch_size=b_size, verbose=1)
    # kekka = model.predict_proba(test, batch_size=b_size, verbose=1)

    with open(args.output_file, 'w') as p:
        for i in kekka:
            p.write("{}\n".format(str(i[0])))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--svm_test_file', help='テストデータ', required=True)
    parser.add_argument('-m', '--model', help='モデルデータ', required=True)
    parser.add_argument('-o', '--output_file', default='output', help='結果出力ファイル')
    args = parser.parse_args()

    main()
