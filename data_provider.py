import pandas as pd
import os
import json

train_label_file = 'ASVspoof2017_train.trn.txt'
dev_label_file = 'ASVspoof2017_dev.trl.txt'
test_label_file = 'ASVspoof2017_eval_v2_key.trl.txt'


class DataProvider(object):
    def __init__(self, feature_name):
        assert feature_name in ['MFCC']
        self._feature_name = feature_name

    def get_train_data(self):
        feature_file = os.path.join('data/features', self._feature_name, 'train_data.json')
        with open(feature_file) as fid:
            data = json.load(fid)
        return data

    def get_data(self, phase):
        assert phase in ['train', 'dev', 'test']
        feature_file = os.path.join('data/features', self._feature_name, '{}.json'.format(phase))
        with open(feature_file) as fid:
            data = json.load(fid)

        file_dir = 'data/ASVspoof2017_{}'.format(phase)
        filenames = os.listdir(file_dir)

        if phase == 'train':
            label_file = train_label_file
        elif phase == 'dev':
            label_file = dev_label_file
        elif phase == 'test':
            label_file = test_label_file
        df = pd.read_csv(os.path.join('data/labels', label_file),
                         sep=' ',
                         header=None,
                         index_col=0)
        labels = []
        for filename in filenames:
            if df[1][filename] == 'genuine':
                labels.append(0)
            elif df[1][filename] == 'spoof':
                labels.append(1)
        return data, labels, filenames

    def next_batch(self, batch_size, phase):
        """

        :param batch_size:
        :param phase: train, dev or test
        :return:
        """
        raise NotImplementedError

if __name__ == '__main__':
    provider = DataProvider('MFCC')
    data = provider.get_data('train')