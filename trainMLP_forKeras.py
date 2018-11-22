# -*- coding: utf-8 -*-

import argparse
from sklearn.datasets import load_svmlight_files

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

import keras.backend as K


def main():
    train, trainTag = load_svmlight_files([args.train])
    train = train.toarray()

    model = create_model(train.shape[-1])

    model.fit(train, trainTag, batch_size=256, epochs=20,
              verbose=1, validation_split=0.3, shuffle=True, initial_epoch=0)

    model.save(f'{args.model_name}.h5', include_optimizer=False)


def fmeasure(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * precision * recall / (precision + recall)


# モデルを設計
def create_model(max_features):
    model = Sequential()

    model.add(Dense(max_features, input_dim=max_features, activation='relu'))

    lap = 3
    for var in range(0, lap):
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))

    for var in range(0, lap):
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))

    for var in range(0, lap):
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))

    for var in range(0, lap):
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=[fmeasure])

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='深層学習を行うプログラム')
    parser.add_argument('-t', '--svm_train_file', help='学習データ', required=True)
    parser.add_argument('-m', '--model_name', default='model', help='モデル名')
    args = parser.parse_args()
    main()
