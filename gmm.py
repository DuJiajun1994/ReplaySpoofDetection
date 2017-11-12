import json
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from data_provider import DataProvider

num_components = 512


def _train_mixture(x, label):
    mixture = GaussianMixture(n_components=num_components)
    mixture.fit(x)
    weights = mixture.weights_.tolist()
    means = mixture.means_.tolist()
    covariances = mixture.covariances_.tolist()
    precisions = mixture.precisions_.tolist()
    params = {
        'weights': weights,
        'means': means,
        'covariances': covariances,
        'precisions': precisions
    }
    with open('output/mixture{}_params.json'.format(label), 'w') as fid:
        json.dump(params, fid)


def _load_mixture(label):
    with open('output/mixture{}_params.json'.format(label)) as fid:
        params = json.load(fid)
    weights = params['weights']
    means = params['means']
    precisions = params['precisions']
    mixture = GaussianMixture(n_components=num_components,
                              weights_init=weights,
                              means_init=means,
                              precisions_init=precisions)
    return mixture


def train_model(provider):
    data = provider.get_train_data()
    _train_mixture(data[0], 0)
    _train_mixture(data[1], 1)


def validate_model(provider):
    data, labels, filenames = provider.get_data('dev')
    mixture0 = _load_mixture(0)
    mixture1 = _load_mixture(1)
    df = pd.DataFrame(columns=['filename', 'label', 'score_0', 'score_1'],
                      dtype={
                          'filename': np.str,
                          'label': np.int,
                          'score_0': np.float,
                          'score_1': np.float
                      })
    for i in range(len(data)):
        score0 = mixture0.score(data[i])
        score1 = mixture1.score(data[i])
        df.loc[i] = [filenames[i], labels[i], score0, score1]
    save_file = 'output/dev_result.csv'
    df.to_csv(save_file, index=False)


def test_model(provider):
    pass


if __name__ == '__main__':
    provider = DataProvider('MFCC')
    train_model(provider)
    validate_model(provider)
    test_model(provider)
