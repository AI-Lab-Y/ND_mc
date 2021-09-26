from os import urandom
from re import A, X
import numpy as np
import random

r = 136 # r = 136 bytes
w = 64 # 1600 / 25 = 64
bytes_per_lane = 8 # 64 / 8 = 8
digest = 32 # digest = 32 bytes

R_CONS = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14]
]
R_CONS = np.array(R_CONS, dtype=np.uint8).reshape((5, 5, 1))

RC = [
    0x8000000000000000, 0x4101000000000000, 0x5101000000000001, 0x1000100000001, 
    0xd101000000000000, 0x8000000100000000, 0x8101000100000001, 0x9001000000000001, 
    0x5100000000000000, 0x1100000000000000, 0x9001000100000000, 0x5000000100000000,
    0xd101000100000000, 0xd100000000000001, 0x9101000000000001, 0xc001000000000001,
    0x4001000000000001, 0x100000000000001, 0x5001000000000000, 0x5000000100000001,
    0x8101000100000001, 0x101000000000001, 0x8000000100000000, 0x1001000100000001
]
RC = np.array(RC, dtype=np.uint64)


def ror(x, l):
    return (x >> l) | (x << (w - l))


def theta(A):
    C = A[:,0] ^ A[:,1] ^ A[:,2] ^ A[:,3] ^ A[:,4]
    index_x = np.array(range(5))
    D = C[(index_x + 4) % 5] ^ ror(C[(index_x + 1) % 5], 1)
    return A ^ D[:, np.newaxis]


def rho(A):
    return ror(A, R_CONS)


def pi(A):
    y = np.array([[0, 1, 2, 3, 4]] * 5)
    x = np.transpose(y)
    return A[(x + 3 * y) % 5, x]


def chi(A):
    index_x = np.array(range(5))
    return A ^ ((~A[(index_x + 1) % 5]) & A[(index_x + 2) % 5])


def f(A, r):
    A = theta(A)
    A = rho(A)
    A = pi(A)
    A = chi(A)
    A[0,0] ^= RC[r]
    return A


def absorb(A, input):
    x, y = 0, 0
    for pos in range(0, r, bytes_per_lane):
        t = np.zeros(input.shape[1], dtype=np.uint64)
        for k in range(pos, pos + bytes_per_lane):
            t = (t << np.uint8(8)) | input[k]
        A[x, y] ^= t
        x += 1
        if x == 5:
            x = 0
            y += 1


def squeeze(A):
    output = []
    x, y = 0, 0
    for pos in range(0, r, bytes_per_lane):
        lane = A[x, y]
        buffer = []
        for _ in range(bytes_per_lane):
            buffer.append(lane & np.uint8(0xff))
            lane = lane >> np.uint8(8)
        buffer.reverse()
        output = output + buffer
        x += 1
        if x == 5:
            x = 0
            y += 1
    output = np.array(output, dtype=np.uint8)
    return output


def pad(message):
    bit_string = message.copy()
    bit_string += [0, 1]
    m = len(bit_string)
    x = r * 8
    j = -1 * m - 2
    while j < 0:
        j += x
    bit_string.append(1)
    bit_string += [0 for _ in range(j)]
    bit_string.append(1)
    assert len(bit_string) % x == 0
    return bit_string


def bit_to_byte(bit_string):
    assert len(bit_string) % 8 == 0
    byte_array = []
    for pos in range(0, len(bit_string), 8):
        tmp = 0
        for j in range(pos, pos + 8):
            tmp = (tmp << 1) | bit_string[j]
        byte_array.append(tmp)
    return np.array(byte_array, dtype=np.uint8)


# plaintext.shape = (n, N * r) = (n, N * 136)
# return: ciphertext.shape = (n, 32)
def SHA3_run(plaintext, Nr):
    plaintext = np.transpose(plaintext)
    assert len(plaintext) % r == 0
    plain_length = len(plaintext)
    plaintext = np.reshape(plaintext, (plain_length, -1))
    A = np.zeros((5, 5, plaintext.shape[1]), dtype=np.uint64)
    for pos in range(0, plain_length, r):
        absorb(A, plaintext[pos:(pos + r)])
        for i in range(Nr):
            A = f(A, i)
    ciphertext = squeeze(A)
    ciphertext = np.transpose(ciphertext[:digest])
    return ciphertext


def convert_to_binary(c0, c1):
    n = len(c0)
    cb0 = np.zeros((n, 256), dtype=np.uint8)
    cb1 = np.zeros((n, 256), dtype=np.uint8)
    byte_pos = 0
    for pos in range(0, 256, 8):
        byte0 = c0[:, byte_pos].copy()
        byte1 = c1[:, byte_pos].copy()
        for k in range(pos + 7, pos - 1, -1):
            cb0[:, k] = byte0 & 1
            cb1[:, k] = byte1 & 1
            byte0 = byte0 >> 1
            byte1 = byte1 >> 1
        byte_pos += 1
    # cb0_tmp = np.zeros((n, 256), dtype=np.uint8)
    # cb1_tmp = np.zeros((n, 256), dtype=np.uint8)
    # for base in range(0, 256, 64):
    #     for i in range(64):
    #         cb0_tmp[:, base + i] = cb0[:, base + 63 - i]
    #         cb1_tmp[:, base + i] = cb1[:, base + 63 - i]
    return np.concatenate((cb0, cb1), axis=1)


def make_target_diff_samples(n=10**7, Nr=2, diff_type=1, diff=[(135, 0x80)]):
    p0 = np.frombuffer(urandom(n*r), dtype=np.uint8).reshape(n, r)
    if diff_type == 1:
        p1 = p0.copy()
        for d in diff:
            p1[:, d[0]] ^= d[1]
    else:
        p1 = np.frombuffer(urandom(n*r), dtype=np.uint8).reshape(n, r)
    c0 = SHA3_run(p0, Nr)
    c1 = SHA3_run(p1, Nr)
    return convert_to_binary(c0, c1)


def make_dataset(n, Nr, diff):
    num = n // 2
    X_p = make_target_diff_samples(num, Nr, 1, diff)
    X_n = make_target_diff_samples(num, Nr, 0, diff)
    Y_p = [1 for _ in range(num)]
    Y_n = [0 for _ in range(num)]
    X = np.concatenate((X_p, X_n), axis=0)
    Y = np.concatenate((Y_p, Y_n))
    return X, Y


# verify  the correctness of our Keccak algorithm
# SHA-3 functions are available at the examples page:
# https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines/example-values
def verify(message, output):
    bit_string = pad(message)
    plaintext = bit_to_byte(bit_string)
    ciphertext = SHA3_run(plaintext, 24).reshape(32)
    res = np.zeros(32, dtype=np.uint8)
    for _ in range(8):
        res = (res << 1) | (ciphertext & 1)
        ciphertext = ciphertext >> 1
    print('digest:')
    for i in res:
        print(hex(i), end=' ')
    print('')
    if np.sum(res == output) == 32:
        print('Keccak is implemented correctly')


if __name__ == '__main__':
    # 30-bit message
    print('30-bit message')
    message = [1,1,0,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,0,1,1,0]
    output = np.array([0xc8,0x24,0x2f,0xef,0x40,0x9e,0x5a,0xe9,0xd1,0xf1,0xc8,0x57,0xae,0x4d,0xc6,0x24,0xb9,0x2b,0x19,0x80,0x9f,0x62,0xaa,0x8c,0x07,0x41,0x1c,0x54,0xa0,0x78,0xb1,0xd0], dtype=np.uint8)
    verify(message, output)

    # 0-bit message
    print('0-bit message')
    message = []
    output = np.array([0xa7,0xff,0xc6,0xf8,0xbf,0x1e,0xd7,0x66,0x51,0xc1,0x47,0x56,0xa0,0x61,0xd6,0x62,0xf5,0x80,0xff,0x4d,0xe4,0x3b,0x49,0xfa,0x82,0xd8,0x0a,0x4b,0x80,0xf8,0x43,0x4a], dtype=np.uint8)
    verify(message, output)

    # 5-bit message
    print('5-bit message')
    message = [1,1,0,0,1]
    output = np.array([0x7b,0x00,0x47,0xcf,0x5a,0x45,0x68,0x82,0x36,0x3c,0xbf,0x0f,0xb0,0x53,0x22,0xcf,0x65,0xf4,0xb7,0x05,0x9a,0x46,0x36,0x5e,0x83,0x01,0x32,0xe3,0xb5,0xd9,0x57,0xaf], dtype=np.uint8)
    verify(message, output)
