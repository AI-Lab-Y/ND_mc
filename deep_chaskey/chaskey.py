import numpy as np
from os import urandom
from copy import deepcopy
import subprocess


def WORD_SIZE():
    return(32)


MASK_VAL = 2 ** WORD_SIZE() - 1


def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))


def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))


def top(p):
    va, vb, vc, vd = p[0], p[1], p[2], p[3]
    tp1 = rol(vb, 5)
    tp2 = va + vb
    tp3 = vc + vd
    tp4 = rol(vd, 8)
    va = tp3
    vb = tp1 ^ tp2
    vc = rol(tp2, 16)
    vd = tp3 ^ tp4
    return(va,vb,vc,vd)


def down(p):
    va, vb, vc, vd = p[0], p[1], p[2], p[3]
    tp1 = rol(vb, 7)
    tp2 = va + vb
    tp3 = vc + vd
    tp4 = rol(vd, 13)
    va = tp3
    vb = tp1 ^ tp2
    vc = rol(tp2, 16)
    vd = tp3 ^ tp4
    return(va,vb,vc,vd)


def permutation_one_round(p):
    va, vb, vc, vd = p[0], p[1], p[2], p[3]
    va, vb, vc, vd = top((va, vb, vc, vd))
    va, vb, vc, vd = down((va, vb, vc, vd))
    return(va, vb, vc, vd)


def permutation_one_round_inverse(p):
    va, vb, vc, vd = p[0], p[1], p[2], p[3]
    va, vb, vc, vd = down((va, vb, vc, vd))
    va, vb, vc, vd = top((va, vb, vc, vd))
    return(va, vb, vc, vd)


def permutation(p, x, y, head=0):   # x代表整数轮，y为0，代表还有半轮;  head = 0, top 开始，否则 down开始
    va, vb, vc, vd = p[0], p[1], p[2], p[3]
    if head == 0:
        if x != 0:
            for i in range(x):
                va, vb, vc, vd = permutation_one_round((va, vb, vc, vd))
        if y != 0:
            va, vb, vc, vd = top((va, vb, vc, vd))
    else:
        if x != 0:
            for i in range(x):
                va, vb, vc, vd = permutation_one_round_inverse((va, vb, vc, vd))
        if y != 0:
            va, vb, vc, vd = down((va, vb, vc, vd))
    return(va, vb, vc, vd)


def timesTwo(key):
    k0, k1, k2, k3 = key[0], key[1], key[2], key[3]
    tp = deepcopy(k0)
    k0 = (k0 << 1) | ((k1 >> 31) & 1)
    k1 = (k1 << 1) | ((k2 >> 31) & 1)
    k2 = (k2 << 1) | ((k3 >> 31) & 1)
    k3 = k3 << 1

    k3[ror(tp, 31) & 1 == 1] = k3[ror(tp, 31) & 1 == 1] ^ 0x00000087
    return(k0, k1, k2, k3)


def convert_to_binary(arr):
    X = np.zeros((8 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(8 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return(X)


def make_target_diff_dataset(n, x, y, head, group_size, diff=(0, 0, 0, 0), type=1):
    key_num = n // group_size
    ka = np.frombuffer(urandom(4 * key_num), dtype=np.uint32)
    kb = np.frombuffer(urandom(4 * key_num), dtype=np.uint32)
    kc = np.frombuffer(urandom(4 * key_num), dtype=np.uint32)
    kd = np.frombuffer(urandom(4 * key_num), dtype=np.uint32)
    ka = np.repeat(ka, group_size)
    kb = np.repeat(kb, group_size)
    kc = np.repeat(kc, group_size)
    kd = np.repeat(kd, group_size)
    vka, vkb, vkc, vkd = timesTwo((ka, kb, kc, kd))

    pa = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    pb = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    pc = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    pd = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    if type == 1:
        na, nb, nc, nd = pa ^ diff[0], pb ^ diff[1], pc ^ diff[2], pd ^ diff[3]
    else:
        na = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        nb = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        nc = np.frombuffer(urandom(4 * n), dtype=np.uint32)
        nd = np.frombuffer(urandom(4 * n), dtype=np.uint32)

    pa, pb, pc, pd = pa ^ ka ^ vka, pb ^ kb ^ vkb, pc ^ kc ^ vkc, pd ^ kd ^ vkd
    tpa, tpb, tpc, tpd = permutation((pa, pb, pc, pd), x, y, head=head)
    # tpa, tpb, tpc, tpd = tpa ^ vka, tpb ^ vkb, tpc ^ vkc, tpd ^ vkd

    na, nb, nc, nd = na ^ ka ^ vka, nb ^ kb ^ vkb, nc ^ kc ^ vkc, nd ^ kd ^ vkd
    fpa, fpb, fpc, fpd = permutation((na, nb, nc, nd), x, y, head=head)
    # fpa, fpb, fpc, fpd = fpa ^ vka, fpb ^ vkb, fpc ^ vkc, fpd ^ vkd

    return(tpa, tpb, tpc, tpd, fpa, fpb, fpc, fpd)


def make_chaskey_dataset(n, x, y, head, group_size, diff=(0, 0, 0, 0)):
    mean = n // 2
    t0a, t0b, t0c, t0d, t1a, t1b, t1c, t1d = make_target_diff_dataset(mean, x, y, head,
                                                                      group_size, diff=diff, type=1)
    f0a, f0b, f0c, f0d, f1a, f1b, f1c, f1d = make_target_diff_dataset(mean, x, y, head,
                                                                      group_size, diff=diff, type=0)

    ct0a = np.concatenate((t0a, f0a), 0)
    ct0b = np.concatenate((t0b, f0b), 0)
    ct0c = np.concatenate((t0c, f0c), 0)
    ct0d = np.concatenate((t0d, f0d), 0)
    ct1a = np.concatenate((t1a, f1a), 0)
    ct1b = np.concatenate((t1b, f1b), 0)
    ct1c = np.concatenate((t1c, f1c), 0)
    ct1d = np.concatenate((t1d, f1d), 0)

    Y1 = [1 for i in range(mean // group_size)]
    Y0 = [0 for i in range(mean // group_size)]
    Y = np.concatenate((Y1, Y0))

    X = convert_to_binary([ct0a, ct0b, ct0c, ct0d, ct1a, ct1b, ct1c, ct1d])
    X = X.reshape((n // group_size, -1))

    index = np.arange((mean // group_size)*2)
    np.random.shuffle(index)
    X = X[index, :]
    Y = Y[index]

    return(X, Y)


def create_train_test_dataset(n1, n2, x, y, head=0, group_size=1, diff=(0, 0, 0, 1)):
    X, Y = make_chaskey_dataset(n1, x, y, head, group_size, diff=diff)
    X_eval, Y_eval = make_chaskey_dataset(n2, x, y, head, group_size, diff=diff)
    return X, Y, X_eval, Y_eval


def uint32_to_byte(p):
    tag = []
    n = len(p)
    for i in reversed(p):
        for _ in range(4):
            tag.append(i & 0xff)
            i = i >> 8
    return np.squeeze(np.array(tag, dtype=np.uint8))


# verify the correctness of Chaskey implementation
def verify(p, k, t):
    ka = np.array([k[0]], dtype=np.uint32)
    kb = np.array([k[1]], dtype=np.uint32)
    kc = np.array([k[2]], dtype=np.uint32)
    kd = np.array([k[3]], dtype=np.uint32)
    vka, vkb, vkc, vkd = timesTwo((ka, kb, kc, kd))
    pa = np.array([p[0]], dtype=np.uint32)
    pb = np.array([p[1]], dtype=np.uint32)
    pc = np.array([p[2]], dtype=np.uint32)
    pd = np.array([p[3]], dtype=np.uint32)
    pa, pb, pc, pd = pa ^ ka ^ vka, pb ^ kb ^ vkb, pc ^ kc ^ vkc, pd ^ kd ^ vkd
    tpd, tpc, tpb, tpa = permutation((pd, pc, pb, pa), 8, 0, 0)
    tpa, tpb, tpc, tpd = tpa ^ vka, tpb ^ vkb, tpc ^ vkc, tpd ^ vkd
    if tpa[0] == t[0] and tpb[0] == t[1] and tpc[0] == t[2] and tpd[0] == t[3]:
        print('chaskey is implemented correctly.')


if __name__ == '__main__':
    # create_train_test_dataset(10**7, 10**6, 4, 0, head=0, diff=(0x8400, 0x0400, 0, 0))
    # create_train_test_dataset(10**7, 10**6, 4, 0, head=0, diff=(0, 0, 0, 1))

    # Correct C version implementation of Chaskey algorithm can be found at https://mouha.be/wp-content/uploads/chaskey-speed.c.
    # This is a (message, key, tag) sample got from chaskey-speed.c.
    pa, pb, pc, pd = 0x0f0e0d0c, 0x0b0a0908, 0x07060504, 0x03020100
    ka, kb, kc, kd = 0x417ACF39, 0x2398E64F, 0x009F389F, 0x833D3433
    ta, tb, tc, td = 0x49831cad, 0x81ca474e, 0xd66a1c71, 0x79271ca9
    verify((pa, pb, pc, pd), (ka, kb, kc, kd), (ta, tb, tc, td))