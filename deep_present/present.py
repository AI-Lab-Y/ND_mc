import numpy as np
import copy
from os import urandom

# for encryption
Sbox = np.array([0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2])

raw_P = [0,  16, 32, 48, 1,  17, 33, 49, 2,  18, 34, 50, 3,  19, 35, 51,
     4,  20, 36, 52, 5,  21, 37, 53, 6,  22, 38, 54, 7,  23, 39, 55,
     8,  24, 40, 56, 9,  25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
     12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]
raw_P = np.array(raw_P)

# Big-Edian
index = np.array([63 - i for i in range(64)])
raw_P = 63 - raw_P[index]

P = np.array([np.where(raw_P == i) for i in range(64)])
P = np.squeeze(P)

# for decryption, to be test
Sbox_inverse = np.array([0x5, 0xe, 0xf, 0x8, 0xc, 0x1, 0x2, 0xd, 0xb, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xa])
P_inverse = raw_P

# for updating keys
KP = np.array([(i+61) % 80 for i in range(80)])


# x shape: (-1, 4)
def get_Sbox_output_enc(x):
    n, m = np.shape(x)
    assert m == 4
    x_val = x[:, 0] * 8 + x[:, 1] * 4 + x[:, 2] * 2 + x[:, 3]
    y_val = Sbox[x_val]
    output = np.zeros((n, 4), dtype=np.uint8)
    for i in range(4):
        output[:, i] = (y_val >> (3 - i)) & 1
    # print('y_val shape is ', np.shape(y_val))
    return output


# x shape: (-1, 4)
def get_Sbox_output_dec(x):
    n, m = np.shape(x)
    assert m == 4
    x_val = x[:, 0] * 8 + x[:, 1] * 4 + x[:, 2] * 2 + x[:, 3]
    y_val = Sbox_inverse[x_val]
    output = np.zeros((n, 4), dtype=np.uint8)
    for i in range(4):
        output[:, i] = (y_val >> (3 - i)) & 1
    # print('y_val shape is ', np.shape(y_val))
    return output


# keys shape: (-1, 80)
def update_master_key(keys, round_counter):
    tp = keys[:, KP]
    new_keys = copy.deepcopy(tp)
    new_keys[:, :4] = get_Sbox_output_enc(tp[:, :4])
    round_counter_arr = np.array([(round_counter >> (4-i)) & 1 for i in range(5) ], dtype=np.uint8)
    new_keys[:, 60:65] = tp[:, 60:65] ^ round_counter_arr
    return new_keys


# keys shape: (-1, 80)
def expand_key(keys, nr):
    n, m = np.shape(keys)
    assert m == 80
    ks = np.zeros((nr+1, n, 64), dtype=np.uint8)
    ks[0] = keys[:, :64]
    for i in range(1, nr+1):
        keys = update_master_key(keys, i)
        ks[i] = keys[:, :64]
    return ks


# x shape: (-1, 64)
def sBoxLayer_enc(x):
    n, m = np.shape(x)
    assert m == 64
    output = np.zeros((n, 64), dtype=np.uint8)
    for i in range(16):
        st = 4 * i
        output[:, st:st+4] = get_Sbox_output_enc(x[:, st:st+4])
    return output


# x shape: (-1, 64)
def sBoxLayer_dec(x):
    n, m = np.shape(x)
    assert m == 64
    output = np.zeros((n, 64), dtype=np.uint8)
    for i in range(16):
        st = 4 * i
        output[:, st:st+4] = get_Sbox_output_dec(x[:, st:st+4])
    return output


# x shape: (-1, 64)
def pLayer_enc(x):
    output = x[:, P]
    return output


# x shape: (-1, 64)
def pLayer_dec(x):
    output = x[:, P_inverse]
    return output


# x shape: (-1, 64)
# subkeys shape: (-1, 64)
def enc_one_round(x, subkeys):
    y = sBoxLayer_enc(x)
    z = pLayer_enc(y)
    output = z ^ subkeys
    return output


# x shape: (-1, 64)
# subkeys shape: (-1, 64)
def dec_one_round(x, subkeys):
    y = pLayer_dec(x)
    z = sBoxLayer_dec(y)
    output = z ^ subkeys
    return output


# x shape: (-1, 64)
# ks shape: (nr, -1, 64)
def encrypt(x, ks):
    nr = ks.shape[0]
    y = x ^ ks[0]
    for i in range(1, nr):
        y = enc_one_round(y, ks[i])
    return y


# x shape: (-1, 64)
# ks shape: (nr, -1, 64)
def decrypt(x, ks):
    nr = ks.shape[0]
    y = x ^ ks[nr-1]
    for i in range(1, nr):
        y = dec_one_round(y, ks[nr - 1 - i])
    return y


def make_target_diff_samples(n=10**7, nr=3, diff_type=1, diff=0x9, return_keys=0):
    x0 = np.frombuffer(urandom(n * 8), dtype=np.uint64)  # .reshape(-1, 1)
    if diff_type == 1:
        x1 = x0 ^ diff
    else:
        x1 = np.frombuffer(urandom(n * 8), dtype=np.uint64)  # .reshape(-1, 1)
    p0 = np.zeros((n, 64), dtype=np.uint8)
    p1 = np.zeros((n, 64), dtype=np.uint8)
    for i in range(64):
        off = 63 - i
        p0[:, i] = (x0 >> off) & 1
        p1[:, i] = (x1 >> off) & 1

    master_keys = np.frombuffer(urandom(n * 80), dtype=np.uint8).reshape(-1, 80) & 1
    subkeys = expand_key(master_keys, nr)
    c0 = encrypt(p0, subkeys)
    c1 = encrypt(p1, subkeys)
    X = np.concatenate((c0, c1), axis=1)

    if return_keys == 1:
        return X, subkeys
    else:
        return X


def make_dataset_with_group_size(n, nr, diff=0x9, group_size=2):
    assert n % group_size == 0
    num = n // 2
    X_p = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff, return_keys=0)
    X_n = make_target_diff_samples(n=num, nr=nr, diff_type=0, return_keys=0)
    X_raw = np.concatenate((X_p, X_n), axis=0)
    n, m = np.shape(X_raw)
    X = X_raw.reshape(-1, group_size * m)
    Y_p = [1 for i in range(num // group_size)]
    Y_n = [0 for i in range(num // group_size)]
    Y = np.concatenate((Y_p, Y_n))
    return X, Y


def verify(n, nr=31):
    x = np.frombuffer(urandom(n * 8), dtype=np.uint64)  # .reshape(-1, 1)
    p = np.zeros((n, 64), dtype=np.uint8)
    for i in range(64):
        off = 63 - i
        p[:, i] = (x >> off) & 1

    master_keys = np.frombuffer(urandom(80), dtype=np.uint8).reshape(-1, 80) & 1
    subkeys = expand_key(master_keys, nr)
    c = encrypt(p, subkeys)
    d = decrypt(c, subkeys)
    print(p ^ d)


# verify(n=10, nr=31)
