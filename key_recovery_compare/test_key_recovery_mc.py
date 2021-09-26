import speck_mc as sp
import numpy as np
from keras.models import model_from_json, load_model
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2


WORD_SIZE = sp.WORD_SIZE()
neutral13 = [22, 21, 20, 14, 15,  7, 23, 30,  0, 24,  8, 31,  1]


def get_reuse_index(key_len=2**16, id_range=32, stride=1):
  idx = []
  for i in range(id_range):
    idx.append(i)
    idx.append((i+stride)%id_range)

  # for data_reuse
  leng = len(idx)
  index = [id_range * i + idx[j] for i in range(key_len) for j in range(leng)]

  return index


index_gen_samples = get_reuse_index(key_len=1, id_range=64, stride=1)


# load distinguishers
mc_net7 = load_model('./saved_model/7_2_mc_distinguisher.h5')
mc_net6 = load_model('./saved_model/6_2_mc_distinguisher.h5')


# test wrong-key decryption
def wrong_key_decryption(n, diff=(0x0040, 0x0), nr=7, net=mc_net7):
  means = np.zeros(2 ** 16);  sig = np.zeros(2 ** 16)

  for i in range(2**16):
    print(i)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys = np.repeat(keys, 2, axis=1)  # sam_len = 2
    ks = sp.expand_key(keys, nr + 1)  # ks[nr-1] = 17123;

    pt0a = np.frombuffer(urandom(2 * n * 2), dtype=np.uint16)
    pt1a = np.frombuffer(urandom(2 * n * 2), dtype=np.uint16)
    pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1]
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks)
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks)

    rsubkeys = i ^ ks[nr]
    c0a, c1a = sp.dec_one_round((ct0a, ct1a), rsubkeys)
    c0b, c1b = sp.dec_one_round((ct0b, ct1b), rsubkeys)
    X = sp.convert_to_binary([c0a, c1a, c0b, c1b])
    # net input
    X = X.reshape((n, 2, 64))

    Z = net.predict(X, batch_size=10000)
    Z = Z.flatten()
    means[i] = np.mean(Z)
    sig[i] = np.std(Z)

  return(means, sig)


# print('begin:')
# mean, std = wrong_key_decryption(n=2000, nr=7, net=mc_net7)
# np.save('./data_wrong_key_mean_7r.npy', mean)
# np.save('./data_wrong_key_std_7r.npy', std)
# print('end')

# mean, std = wrong_key_decryption(n=2000, nr=6, net=mc_net6)
# np.save('./data_wrong_key_mean_6r.npy', mean)
# np.save('./data_wrong_key_std_6r.npy', std)


mc_m7 = np.load('./mean_std/mc/2_mc/data_wrong_key_mean_7r.npy')
mc_s7 = np.load('./mean_std/mc/2_mc/data_wrong_key_std_7r.npy'); mc_s7 = 1.0/mc_s7
mc_m6 = np.load('./mean_std/mc/2_mc/data_wrong_key_mean_6r.npy')
mc_s6 = np.load('./mean_std/mc/2_mc/data_wrong_key_std_6r.npy'); mc_s6 = 1.0/mc_s6


# binarize a given ciphertext sample
# ciphertext is given as a sequence of arrays
# each array entry contains one word of ciphertext for all ciphertexts given
def convert_to_binary(l):
  n = len(l)
  k = WORD_SIZE * n
  X = np.zeros((k, len(l[0])), dtype=np.uint8)

  for i in range(k):
    index = i // WORD_SIZE
    offset = WORD_SIZE - 1 - i % WORD_SIZE
    X[i] = (l[index] >> offset) & 1
  X = X.transpose()
  return(X)


def hw(v):
  res = np.zeros(v.shape, dtype=np.uint8)
  for i in range(16):
    res = res + ((v >> i) & 1)
  return(res)


low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16)
low_weight = low_weight[hw(low_weight) <= 2]


# make a plaintext structure
# takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits
def make_structure(pt0, pt1, diff=(0x211, 0xa04), neutral_bits=[20, 21, 22, 14, 15]):
  p0 = np.copy(pt0); p1 = np.copy(pt1)
  p0 = p0.reshape(-1, 1); p1 = p1.reshape(-1, 1)

  for i in neutral_bits:
    d = 1 << i; d0 = d >> 16; d1 = d & 0xffff
    p0 = np.concatenate([p0, p0 ^ d0], axis=1)
    p1 = np.concatenate([p1, p1 ^ d1], axis=1)
  p0b = p0 ^ diff[0]; p1b = p1 ^ diff[1]
  return(p0, p1, p0b, p1b)


# generate a Speck key, return expanded key
def gen_key(nr):
  key = np.frombuffer(urandom(8), dtype=np.uint16)
  ks = sp.expand_key(key, nr)
  return(ks)


def gen_plain(n):
  pt0 = np.frombuffer(urandom(2*n), dtype=np.uint16)
  pt1 = np.frombuffer(urandom(2*n), dtype=np.uint16)
  return(pt0, pt1)


def gen_challenge(n, nr, diff=(0x211, 0xa04), neutral_bits=[20, 21, 22, 14, 15, 23], keyschedule='real'):
  pt0, pt1 = gen_plain(n)
  pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
  pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), 0)
  pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), 0)
  key = gen_key(nr)

  if (keyschedule is 'free'): key = np.frombuffer(urandom(2*nr), dtype=np.uint16)

  ct0a, ct1a = sp.encrypt((pt0a, pt1a), key)
  ct0b, ct1b = sp.encrypt((pt0b, pt1b), key)

  # apply data reuse strategy
  # print('ct0a shape is ', np.shape(ct0a))
  ct0a1 = ct0a[:, index_gen_samples]
  ct1a1 = ct1a[:, index_gen_samples]
  ct0b1 = ct0b[:, index_gen_samples]
  ct1b1 = ct1b[:, index_gen_samples]
  # print('new ct0a shape is ', np.shape(ct0a1))
  return([ct0a, ct1a, ct0b, ct1b],[ct0a1, ct1a1, ct0b1, ct1b1], key)


def find_good(cts, key, nr=3, target_diff=(0x0040, 0x0)):
  pt0a, pt1a = sp.decrypt((cts[0], cts[1]), key[nr:])
  pt0b, pt1b = sp.decrypt((cts[2], cts[3]), key[nr:])
  diff0 = pt0a ^ pt0b; diff1 = pt1a ^ pt1b
  d0 = (diff0 == target_diff[0]); d1 = (diff1 == target_diff[1])
  d = d0 * d1
  v = np.sum(d, axis=1)
  return(v)


# having a good key candidate, exhaustively explore all keys with hamming distance less than two of this key
def verifier_search(cts, best_guess, use_n=64, net=mc_net6):
  ck1 = best_guess[0] ^ low_weight
  ck2 = best_guess[1] ^ low_weight
  n = len(ck1)
  ck1 = np.repeat(ck1, n)
  keys1 = np.copy(ck1)
  ck2 = np.tile(ck2, n)
  keys2 = np.copy(ck2)
  ck1 = np.repeat(ck1, use_n)
  ck2 = np.repeat(ck2, use_n)

  ct0a = np.tile(cts[0][0:use_n], n*n)
  ct1a = np.tile(cts[1][0:use_n], n*n)
  ct0b = np.tile(cts[2][0:use_n], n*n)
  ct1b = np.tile(cts[3][0:use_n], n*n)
  pt0a, pt1a = sp.dec_one_round((ct0a, ct1a), ck1)
  pt0b, pt1b = sp.dec_one_round((ct0b, ct1b), ck1)
  pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), ck2)
  pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), ck2)      #
  X = sp.convert_to_binary([pt0a, pt1a, pt0b, pt1b])    # X shape (n*n*use_n, 64)

  # apply data reuse strategy
  new_x = X.reshape(n * n * use_n // 2, 2, 64)    # since group size = 2, thus, the first dimension is use_n // 2
  Z = net.predict(new_x, batch_size=10000)    # Z shape (n*n*use_n, 1)

  Z = Z / (1 - Z)
  Z = np.log2(Z)
  Z = Z.reshape(-1, use_n // 2)   # since group size = 2, thus, the second dimension is use_n // 2
  v = np.mean(Z, axis=1) * use_n // 2   # since group size = 2, thus, the second dimension is use_n // 2
  m = np.argmax(v); val = v[m]
  key1 = keys1[m]; key2 = keys2[m]

  return(key1, key2, val)


# here, we use some symmetries of the wrong key performance profile
# by performing the optimization step only on the 14 lowest bits and randomizing the others
# on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
# In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here

tmp_br = np.arange(2**14, dtype=np.uint16)
tmp_br = np.repeat(tmp_br, 32).reshape(-1, 32)


def bayesian_rank_kr(cand, emp_mean, m=mc_m7, s=mc_s7):
  global tmp_br

  n = len(cand)
  if (tmp_br.shape[1] != n):
      tmp_br = np.arange(2**14, dtype=np.uint16)
      tmp_br = np.repeat(tmp_br, n).reshape(-1, n)

  tmp = tmp_br ^ cand   # cand是当前推荐密钥，tmp_br是假设的真实密钥

  v = (emp_mean - m[tmp]) * s[tmp]
  v = v.reshape(-1, n)
  scores = np.linalg.norm(v, axis=1)

  return(scores)


def bayesian_key_recovery(cts, net=mc_net7, m=mc_m7, s=mc_s7, num_cand=32, num_iter=5, seed=None):
  # cts shape (4, 64)  n = 64 represents the number of samples
  n = len(cts[0])
  keys = np.random.choice(2**(WORD_SIZE-2), num_cand, replace=False)
  if (not seed is None):
    keys = np.copy(seed)

  ct0a, ct1a, ct0b, ct1b = np.tile(cts[0], num_cand), np.tile(cts[1], num_cand), \
                           np.tile(cts[2], num_cand), np.tile(cts[3], num_cand)   # ct0a shape (n*num_cand,)/(2048,)
  scores = np.zeros(2**(WORD_SIZE-2))
  used = np.zeros(2**(WORD_SIZE-2))
  all_keys = np.zeros(num_cand * num_iter, dtype=np.uint16)
  all_v = np.zeros(num_cand * num_iter)

  for i in range(num_iter):
    k = np.repeat(keys, n)
    c0a, c1a = sp.dec_one_round((ct0a, ct1a), k)
    c0b, c1b = sp.dec_one_round((ct0b, ct1b), k)    # c0a, c1a, c0b, c1b shape (n*num_cand,)/(2048,)
    X = sp.convert_to_binary([c0a, c1a, c0b, c1b])  # [c0a, c1a, c0b, c1b] shape (4, 2048) x shape (2048, 64)

    # apply data reuse strategy
    new_x = X.reshape(num_cand * n // 2, 2, 64)  # since group size = 2, the first dimension is n // 2

    Z = net.predict(new_x, batch_size=10000)
    Z = Z.reshape(num_cand, -1)
    means = np.mean(Z, axis=1)

    Z = Z/(1-Z)
    Z = np.log2(Z)
    v = np.sum(Z, axis=1)

    all_v[i * num_cand:(i+1)*num_cand] = v
    all_keys[i * num_cand:(i+1)*num_cand] = np.copy(keys)
    scores = bayesian_rank_kr(keys, means, m=m, s=s)
    tmp = np.argpartition(scores+used, num_cand)
    keys = tmp[0:num_cand]
    r = np.random.randint(0, 4, num_cand, dtype=np.uint16); r = r << 14; keys = keys ^ r

  return(all_keys, scores, all_v)


def test_bayes(index, cts, it=1, cutoff1=10.0, cutoff2=100.0, net=mc_net7, net_help=mc_net6, m_main=mc_m7, m_help=mc_m6, s_main=mc_s7, s_help=mc_s6, verify_breadth=None):
  # cts shape (4, 100, 64)
  n = len(cts[0])
  if (verify_breadth is None):
      verify_breadth = len(cts[0][0])
  alpha = sqrt(n)
  best_val = -100.0
  best_key = (0, 0)
  best_pod = 0
  bp = 0
  bv = -100.0
  keys = np.random.choice(2**WORD_SIZE, 32, replace=False)
  eps = 0.001
  local_best = np.full(n, -10)
  glob_best = np.full(n, -10)
  num_visits = np.full(n, eps)

  for j in range(it):
      priority = 0.5 * glob_best + 0.5 * local_best + alpha * np.sqrt(log2(j+1) / num_visits)
      i = np.argmax(priority)
      num_visits[i] = num_visits[i] + 1
      if (best_val > cutoff2):
        improvement = (verify_breadth > 0)

        while improvement:
          k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]],
                                        best_key, net=net_help, use_n=verify_breadth)
          improvement = (val > best_val)

          if (improvement):
            best_key = (k1, k2)
            best_val = val

        return(best_key, j)

      keys, scores, v = bayesian_key_recovery([cts[0][i], cts[1][i], cts[2][i], cts[3][i]], num_cand=32,
                                              num_iter=5, net=net, m=m_main, s=s_main)

      vtmp = np.max(v)
      if (vtmp > local_best[i]):
          local_best[i] = vtmp
      if (vtmp > bv):
        bv = vtmp
        bp = i
      if (vtmp > cutoff1):
        l2 = [i for i in range(len(keys)) if v[i] > cutoff1]
        # print('phase pass again, cur pos is ', i, ' first score is ', vtmp)

        for i2 in l2:
          c0a, c1a = sp.dec_one_round((cts[0][i], cts[1][i]), keys[i2])
          c0b, c1b = sp.dec_one_round((cts[2][i], cts[3][i]), keys[i2])
          keys2, scores2, v2 = bayesian_key_recovery([c0a, c1a, c0b, c1b], num_cand=32, num_iter=5,
                                                   m=m_help, s=s_help, net=net_help)
          vtmp2 = np.max(v2)

          # update priority with second rounds reward
          if vtmp2 > local_best[i]:
            glob_best[i] = vtmp2

          if (vtmp2 > best_val):
            best_val = vtmp2
            best_key = (keys[i2], keys2[np.argmax(v2)])
            best_pod=i
            print('new best key is ', best_key, ' second score is ', best_val, ' cur index is ', i)

      if num_visits[i] > 20:    # punish the misleaded samples
        local_best[i], glob_best[i] = -10, -10

  improvement = (verify_breadth > 0)

  while improvement:
    k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]],
                                  best_key, net=net_help, use_n=verify_breadth)
    improvement = (val > best_val)

    if (improvement):
      best_key = (k1, k2); best_val = val

  return(best_key, it)


def test(index, n, ct_all, key_all, nr=11, num_structures=100, it=500, cutoff1=10.0, cutoff2=100.0, neutral_bits=[20,21,22,14,15,23], keyschedule='real',net=mc_net7, net_help=mc_net6, m_main=mc_m7, s_main=mc_s7,  m_help=mc_m6, s_help=mc_s6, verify_breadth=None):

  print("Checking Speck32/64 implementation.")

  if (not sp.check_testvector()):
    print("Error. Aborting.")
    return(0)

  arr1 = np.zeros(n, dtype=np.uint16)
  arr2 = np.zeros(n, dtype=np.uint16)

  data = 0
  av=0.0
  good = np.zeros(n, dtype=np.uint8)
  zkey = np.zeros(nr,dtype=np.uint16)

  tt = np.zeros(n,dtype=float)
  for i in range(n):
    ts = time()
    print("Test:", i)
    ct, key = ct_all[i], key_all[i]
    g = find_good(ct, key); g = np.max(g); good[i] = g
    print('first true key is ', hex(key[nr-1]), '  second true key is ', hex(key[nr-2]))
    guess, num_used = test_bayes(index=index, cts=ct, it=it, cutoff1=cutoff1, cutoff2=cutoff2,
                                 net=net, net_help=net_help, m_main=m_main, s_main=s_main, m_help=m_help,
                                 s_help=s_help, verify_breadth=verify_breadth)
    num_used = min(num_structures, num_used); data = data + 2 * (2 ** len(neutral_bits)) * num_used
    arr1[i] = guess[0] ^ key[nr-1]
    arr2[i] = guess[1] ^ key[nr-2]
    print("Difference between real key and key guess: ", hex(arr1[i]), hex(arr2[i]))
    te = time()
    tt[i] = te - ts

  print("Done.")
  d1 = [hex(x) for x in arr1]
  d2 = [hex(x) for x in arr2]
  print("Differences between guessed and last key:", d1)
  print("Differences between guessed and second-to-last key:", d2)
  print("Data blocks used (average, log2): ", log2(data) - log2(n))
  return(arr1, arr2, good,tt)
