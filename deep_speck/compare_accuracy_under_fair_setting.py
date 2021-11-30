import speck as sp
import numpy as np
import matplotlib.pyplot as plt
from os import urandom
from keras.models import load_model

def make_target_diff_batch_samples(n=10**6, N=8, nr=6, diff=(0x0040, 0)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    keys = np.frombuffer(urandom(8 * n * N), dtype=np.uint16).reshape(4, n, N)
    ks = sp.expand_key(keys, nr)
    plain0l = np.frombuffer(urandom(2 * n * N), dtype=np.uint16).reshape(n, N)
    plain0r = np.frombuffer(urandom(2 * n * N), dtype=np.uint16).reshape(n, N)
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)
    plain1l[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples * N), dtype=np.uint16).reshape(-1, N)
    plain1r[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples * N), dtype=np.uint16).reshape(-1, N)
    ctdata0l, ctdata0r = sp.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = sp.encrypt((plain1l, plain1r), ks)
    ctdata0l = ctdata0l.flatten(); ctdata0r = ctdata0r.flatten(); ctdata1l = ctdata1l.flatten(); ctdata1r = ctdata1r.flatten()
    X = sp.convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]).reshape((n, N, -1))
    return (X, Y)

def sample_identical(X):
    return X

def sample_sequence(X, k):
    n = X.shape[0]
    N = X.shape[1]
    block_size = X.shape[2]
    sample_index = []
    # data reuse
    # step = 1
    for i in range(N):
        for j in range(k):
            sample_index.append((i + j) % N)
    sample_index = np.array(sample_index)
    A = X[:, sample_index, :].reshape((n, -1, k * block_size))
    return A

def sample_unique(X, k):
    n = X.shape[0]
    N = X.shape[1]
    N -= N % k
    block_size = X.shape[2]
    return np.reshape(X[:, :N, :], (n, -1, k * block_size))

def analyse_median(Z):
    return np.median(Z, axis=1)

def analyse_mean(Z):
    return np.mean(Z, axis=1)

def show_distinguisher_batch_acc(n=10**6, N=8, nr=7, diff=(0x40,0), net_path='./', sample_method='identical', k=1, analyse_method='median', X=None, Y=None):
    net = load_model(net_path)
    if X is None:
        X, Y = make_target_diff_batch_samples(n, N, nr, diff)
    if sample_method == 'identical':
        A = sample_identical(X)
    elif sample_method == 'sequence':
        A = sample_sequence(X, k)
    elif sample_method == 'unique':
        A = sample_unique(X, k)
    M = A.shape[1]
    A = np.reshape(A, (n * M, -1))
    Z = net.predict(A, batch_size=10000).reshape(n, M)
    Z = np.log(Z / (1 - Z))
    if analyse_method == 'median':
        S = analyse_median(Z)
    elif analyse_method == 'mean':
        S = analyse_mean(Z)
    acc = np.sum(((Y == 0) & (S < 0)) | ((Y == 1) & (S > 0))) / n
    print('acc is', acc)

if __name__ == '__main__':
    n = 10**6
    nr = 7
    N = 8
    k = 2
    gohr_net_path = './saved_model/{}_distinguisher.h5'.format(nr)
    mc_net_path = './saved_model/mc/{}_{}_mc_distinguisher.h5'.format(nr, k)
    X, Y = make_target_diff_batch_samples(n=n, N=N, nr=nr, diff=(0x40, 0))
    # gohr
    show_distinguisher_batch_acc(n=n, N=N, nr=nr, diff=(0x40, 0), net_path=gohr_net_path, sample_method='identical', k=1, analyse_method='median', X=X, Y=Y)
    # mc without data reuse
    show_distinguisher_batch_acc(n=n, N=N, nr=nr, diff=(0x40, 0), net_path=mc_net_path, sample_method='unique', k=k, analyse_method='median', X=X, Y=Y)
    # mc with data reuse
    show_distinguisher_batch_acc(n=n, N=N, nr=nr, diff=(0x40, 0), net_path=mc_net_path, sample_method='sequence', k=k, analyse_method='median', X=X, Y=Y)