import numpy as np
from os import urandom
from copy import deepcopy


def WORD_SIZE():
    return (32)


MASK_VAL = 2 ** WORD_SIZE() - 1


def rol(x, k):
    return (((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))


def ror(x, k):
    return ((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))


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
    return (va, vb, vc, vd)


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
    return (va, vb, vc, vd)


def permutation_one_round(p):
    va, vb, vc, vd = p[0], p[1], p[2], p[3]
    va, vb, vc, vd = top((va, vb, vc, vd))
    va, vb, vc, vd = down((va, vb, vc, vd))
    return (va, vb, vc, vd)


def permutation_one_round_inverse(p):
    va, vb, vc, vd = p[0], p[1], p[2], p[3]
    va, vb, vc, vd = down((va, vb, vc, vd))
    va, vb, vc, vd = top((va, vb, vc, vd))
    return (va, vb, vc, vd)


def permutation(p, x, y, head=0):  # x代表整数轮，y为0，代表还有半轮;  head = 0, top 开始，否则 down开始
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
    return (va, vb, vc, vd)


def timesTwo(key):
    k0, k1, k2, k3 = key[0], key[1], key[2], key[3]
    tp = deepcopy(k0)
    k0 = rol(k0, 1) | (ror(k1, 31) & 1)
    k1 = rol(k1, 1) | (ror(k2, 31) & 1)
    k2 = rol(k2, 1) | (ror(k3, 31) & 1)
    k3 = rol(k3, 1) | (ror(tp, 31) & 1)

    k3[ror(tp, 31) & 1 == 1] = k3[ror(tp, 31) & 1 == 1] ^ 0x00000087
    return (k0, k1, k2, k3)


def convert_to_binary(arr):
    X = np.zeros((8 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(8 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return (X)


def make_target_diff_dataset(n, x, y, head, diff=(0, 0, 0, 0), type=1):
    ka = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    kb = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    kc = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    kd = np.frombuffer(urandom(4 * n), dtype=np.uint32)
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

    return (tpa, tpb, tpc, tpd, fpa, fpb, fpc, fpd)


def make_chaskey_dataset(n, x, y, head, group_size, diff=(0, 0, 0, 0)):
    mean = n // 2
    t0a, t0b, t0c, t0d, t1a, t1b, t1c, t1d = make_target_diff_dataset(mean, x, y, head, diff=diff, type=1)
    f0a, f0b, f0c, f0d, f1a, f1b, f1c, f1d = make_target_diff_dataset(mean, x, y, head, diff=diff, type=0)

    ct0a = np.concatenate((t0a, f0a), 0)
    ct0b = np.concatenate((t0b, f0b), 0)
    ct0c = np.concatenate((t0c, f0c), 0)
    ct0d = np.concatenate((t0d, f0d), 0)
    ct1a = np.concatenate((t1a, f1a), 0)
    ct1b = np.concatenate((t1b, f1b), 0)
    ct1c = np.concatenate((t1c, f1c), 0)
    ct1d = np.concatenate((t1d, f1d), 0)
    X = convert_to_binary([ct0a, ct0b, ct0c, ct0d, ct1a, ct1b, ct1c, ct1d])
    X = X.reshape((n // group_size, -1))

    Y1 = [1 for _ in range(mean // group_size)]
    Y0 = [0 for _ in range(mean // group_size)]
    Y = np.concatenate((Y1, Y0))

    # index = np.arange((mean // group_size)*2)
    # np.random.shuffle(index)
    # X = X[index, :]
    # Y = Y[index]

    return (X, Y)


def create_train_test_dataset(n1, n2, x, y, head=0, group_size=1, diff=(0, 0, 0, 1)):
    X, Y = make_chaskey_dataset(n1, x, y, head, group_size, diff=diff)
    X_eval, Y_eval = make_chaskey_dataset(n2, x, y, head, group_size, diff=diff)

    return X, Y, X_eval, Y_eval

# if __name__ == '__main__':
# create_train_test_dataset(10**7, 10**6, 4, 0, head=0, diff=(0x8400, 0x0400, 0, 0))
# create_train_test_dataset(10**7, 10**6, 4, 0, head=0, diff=(0, 0, 0, 1))


def make_train_data_with_joint_key(pa, pb, pc, pd, na, nb, nc, nd, x, y, head, group_size=2):
    n = len(pa)
    assert n % group_size == 0
    num = n // group_size

    ka = np.frombuffer(urandom(4 * num), dtype=np.uint32)
    kb = np.frombuffer(urandom(4 * num), dtype=np.uint32)
    kc = np.frombuffer(urandom(4 * num), dtype=np.uint32)
    kd = np.frombuffer(urandom(4 * num), dtype=np.uint32)
    ka = np.repeat(ka, group_size)
    kb = np.repeat(kb, group_size)
    kc = np.repeat(kc, group_size)
    kd = np.repeat(kd, group_size)
    vka, vkb, vkc, vkd = timesTwo((ka, kb, kc, kd))

    pa, pb, pc, pd = pa ^ ka ^ vka, pb ^ kb ^ vkb, pc ^ kc ^ vkc, pd ^ kd ^ vkd
    tpa, tpb, tpc, tpd = permutation((pa, pb, pc, pd), x, y, head=head)
    # tpa, tpb, tpc, tpd = tpa ^ vka, tpb ^ vkb, tpc ^ vkc, tpd ^ vkd

    na, nb, nc, nd = na ^ ka ^ vka, nb ^ kb ^ vkb, nc ^ kc ^ vkc, nd ^ kd ^ vkd
    fpa, fpb, fpc, fpd = permutation((na, nb, nc, nd), x, y, head=head)
    # fpa, fpb, fpc, fpd = fpa ^ vka, fpb ^ vkb, fpc ^ vkc, fpd ^ vkd

    return (tpa, tpb, tpc, tpd, fpa, fpb, fpc, fpd)


def make_train_data_without_joint_key(pa, pb, pc, pd, na, nb, nc, nd, x, y, head):
    n = len(pa)

    ka = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    kb = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    kc = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    kd = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    vka, vkb, vkc, vkd = timesTwo((ka, kb, kc, kd))

    pa, pb, pc, pd = pa ^ ka ^ vka, pb ^ kb ^ vkb, pc ^ kc ^ vkc, pd ^ kd ^ vkd
    tpa, tpb, tpc, tpd = permutation((pa, pb, pc, pd), x, y, head=head)
    # tpa, tpb, tpc, tpd = tpa ^ vka, tpb ^ vkb, tpc ^ vkc, tpd ^ vkd

    na, nb, nc, nd = na ^ ka ^ vka, nb ^ kb ^ vkb, nc ^ kc ^ vkc, nd ^ kd ^ vkd
    fpa, fpb, fpc, fpd = permutation((na, nb, nc, nd), x, y, head=head)
    # fpa, fpb, fpc, fpd = fpa ^ vka, fpb ^ vkb, fpc ^ vkc, fpd ^ vkd

    return (tpa, tpb, tpc, tpd, fpa, fpb, fpc, fpd)


def make_train_data(n, x, y, head=0, group_size=1, diff=(0, 0, 0, 1)):
    assert n % (2 * group_size) == 0
    num = n // 2

    pa = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    pb = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    pc = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    pd = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    na = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    nb = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    nc = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    nd = np.frombuffer(urandom(4 * n), dtype=np.uint32)
    na[:num], nb[:num], nc[:num], nd[:num] = pa[:num] ^ diff[0], pb[:num] ^ diff[1], pc[:num] ^ diff[2], pd[:num] ^ diff[3]

    c0a_j, c0b_j, c0c_j, c0d_j, c1a_j, c1b_j, c1c_j, c1d_j = make_train_data_with_joint_key(pa, pb, pc, pd, na,
                                                                                            nb, nc, nd, x, y, head,
                                                                                            group_size=group_size)

    c0a, c0b, c0c, c0d, c1a, c1b, c1c, c1d = make_train_data_without_joint_key(pa, pb, pc, pd, na, nb,
                                                                               nc, nd, x, y, head)

    # generate train data with joint key
    X_raw_j = convert_to_binary([c0a_j, c0b_j, c0c_j, c0d_j, c1a_j, c1b_j, c1c_j, c1d_j])
    X_j = X_raw_j.reshape((n // group_size, -1))

    X = convert_to_binary([c0a, c0b, c0c, c0d, c1a, c1b, c1c, c1d])
    X = X.reshape((n // group_size, -1))

    Y1 = [1 for _ in range(num // group_size)]
    Y0 = [0 for _ in range(num // group_size)]
    Y = np.concatenate((Y1, Y0))

    return X_j, X, Y