import numpy as np
import scipy.stats as st


def cal_DC(p0=0.5, p1=0.5, p2=0.5, p3=0.5, bp=0.005, bn=0.005):
    z_1_bp = st.norm.ppf(1 - bp)
    z_1_bn = st.norm.ppf(1 - bn)
    mu_p = p0 * p1 + (1 - p0) * p3
    mu_n = p0 * p2 + (1 - p0) * p3
    sig_p = np.sqrt(p0 * p1 * (1 - p1) + (1 - p0) * p3 * (1 - p3))
    sig_n = np.sqrt(p0 * p2 * (1 - p2) + (1 - p0) * p3 * (1 - p3))
    # print('z_1_bp is ', z_1_bp, ' mu_p is ', mu_p, ' sig_p is  ', sig_p)
    # print('z_1_bn is ', z_1_bn, ' mu_n is ', mu_n, ' sig_n is  ', sig_n)
    x = z_1_bp * sig_p + z_1_bn * sig_n
    y = np.abs(mu_p - mu_n)

    N = (x / y) * (x / y)
    dc = np.log2(N)
    print('data complexity is', N)
    print('the weight of data complexity is ', dc)

    # calculate the decision threshold t
    sig = sig_p * np.sqrt(N)
    mu = mu_p * N
    t = mu - sig * z_1_bp
    print('t is ', t)


# deep_speck, 3 + 5 attack, d = 2
# dc = 2**14.212 = 18971, t = 1054
print('deep_speck, 3 + 5 attack, d = 2')
cal_DC(p0=2**(-6), p1=0.8976952, p2=0.2280031, p3=0.0462333, bp=0.005, bn=2**(-16))

# deep_speck, 3 + 6 attack, d = 2
# dc = 2**16.911 = 123209, t = 19147
print('deep_speck, 3 + 6 attack, d = 2')
cal_DC(p0=2**(-6), p1=0.7146867, p2=0.2745045, p3=0.1491974, bp=0.005, bn=2**(-16))

# deep_speck, 3 + 7 attack, d = 2
# dc = 2**20.509 = 1492307, t = 495391
print('deep_speck, 3 + 7 attack, d = 2')
cal_DC(p0=2**(-6), p1=0.5435416, p2=0.3773094, p3=0.3296132, bp=0.005, bn=2**(-16))

# deep_speck, 3 + 5 attack, d = 1
# dc = 2**14.720 = 26990, t = 1517
print('deep_speck, 3 + 5 attack, d = 1')
cal_DC(p0=2**(-6), p1=0.8976952, p2=0.3335031, p3=0.0462333, bp=0.005, bn=2**(-16))

# deep_speck, 3 + 6 attack, d = 1
# dc = 2**17.456 = 179848, t = 28030
print('deep_speck, 3 + 6 attack, d = 1')
cal_DC(p0=2**(-6), p1=0.7146867, p2=0.3499637, p3=0.1491974, bp=0.005, bn=2**(-16))

# deep_speck, 3 + 7 attack, d = 1
# dc = 2**21.081 = 2218438, t = 736836
print('deep_speck, 3 + 7 attack, d = 1')
cal_DC(p0=2**(-6), p1=0.5435416, p2=0.4071833, p3=0.3296132, bp=0.005, bn=2**(-16))

print('')

# deep_speck_mc, 3 + 5 attack, k = 2, d = 2
# dc = 2**13.165 = 9184, t = 272
print('deep_speck_mc, 3 + 5 attack, k = 2, d = 2')
cal_DC(p0=2**(-6), p1=0.9665034, p2=0.326112, p3=0.0185456, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 4, d = 2
# dc = 2**12.901 = 7651, t = 151
print('deep_speck_mc, 3 + 5 attack, k = 4, d = 2')
cal_DC(p0=2**(-6), p1=0.9894372, p2=0.5175412, p3=0.0069172, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 8, d = 2
# dc = 2**12.190 = 4672, t = 71
print('deep_speck_mc, 3 + 5 attack, k = 8, d = 2')
cal_DC(p0=2**(-6), p1=0.9987888, p2=0.680548, p3=0.0007616, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 16, d = 2
# dc = 2**13.107 = 8821, t = 136
print('deep_speck_mc, 3 + 5 attack, k = 16, d = 2')
cal_DC(p0=2**(-6), p1=0.9999072, p2=0.873136, p3=1.92e-05, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 2, d = 1
# dc = 2**13.979 = 16145, t = 494
print('deep_speck_mc, 3 + 5 attack, k = 2, d = 1')
cal_DC(p0=2**(-6), p1=0.9665034, p2=0.4801842, p3=0.0185456, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 4, d = 1
# dc = 2**14.187 = 18651, t = 386
print('deep_speck_mc, 3 + 5 attack, k = 4, d = 1')
cal_DC(p0=2**(-6), p1=0.9894372, p2=0.6927208, p3=0.0069172, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 8, d = 1
# dc = 2**14.085 = 17377, t = 274
print('deep_speck_mc, 3 + 5 attack, k = 8, d = 1')
cal_DC(p0=2**(-6), p1=0.9987888, p2=0.8603952, p3=0.0007616, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 16, d = 1
# dc = 2**15.399 = 43206, t = 673
print('deep_speck_mc, 3 + 5 attack, k = 16, d = 1')
cal_DC(p0=2**(-6), p1=0.9999072, p2=0.9672112, p3=1.92e-05, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 6 attack, k = 2, d = 2
# dc = 2**15.821 = 57895, t = 6104
print('deep_speck_mc, 3 + 6 attack, k = 2, d = 2')
cal_DC(p0=2**(-6), p1=0.8301926, p2=0.2943016, p3=0.0971788, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 6 attack, k = 4, d = 2
# dc = 2**14.764 = 27819, t = 1814
print('deep_speck_mc, 3 + 6 attack, k = 4, d = 2')
cal_DC(p0=2**(-6), p1=0.9228128, p2=0.3221144, p3=0.0551936, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 6 attack, k = 8, d = 2
# dc = 2**14.681 = 26260, t = 1347
print('deep_speck_mc, 3 + 6 attack, k = 8, d = 2')
cal_DC(p0=2**(-6), p1=0.9470632, p2=0.4097384, p3=0.0402856, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 6 attack, k = 16, d = 2
# dc = 2**14.720 = 26984, t = 855
print('deep_speck_mc, 3 + 6 attack, k = 16, d = 2')
cal_DC(p0=2**(-6), p1=0.9765312, p2=0.59896, p3=0.0188624, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 6 attack, k = 2, d = 1
# dc = 2**16.484 = 91702, t = 9729
print('deep_speck_mc, 3 + 6 attack, k = 2, d = 1')
cal_DC(p0=2**(-6), p1=0.8301926, p2=0.4036316, p3=0.0971788, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 6 attack, k = 4, d = 1
# dc = 2**15.606 = 49879, t = 3297
print('deep_speck_mc, 3 + 6 attack, k = 4, d = 1')
cal_DC(p0=2**(-6), p1=0.9228128, p2=0.4729764, p3=0.0551936, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 6 attack, k = 8, d = 1
# dc = 2**15.750 = 55105, t = 2881
print('deep_speck_mc, 3 + 6 attack, k = 8, d = 1')
cal_DC(p0=2**(-6), p1=0.9470632, p2=0.576036, p3=0.0402856, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 6 attack, k = 16, d = 1
# dc = 2**16.361 = 84156, t = 2744
print('deep_speck_mc, 3 + 6 attack, k = 16, d = 1')
cal_DC(p0=2**(-6), p1=0.9765312, p2=0.765712, p3=0.0188624, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 7 attack, k = 2, d = 2
# dc = 2**19.761 = 888254, t = 286907
print('deep_speck_mc, 3 + 7 attack, k = 2, d = 2')
cal_DC(p0=2**(-6), p1=0.599359, p2=0.3855446, p3=0.3199108, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 7 attack, k = 4, d = 2
# dc = 2**18.886 = 484590, t = 142708
print('deep_speck_mc, 3 + 7 attack, k = 4, d = 2')
cal_DC(p0=2**(-6), p1=0.6598372, p2=0.3780524, p3=0.290402, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 7 attack, k = 8, d = 2
# dc = 2**18.744 = 439087, t = 128650
print('deep_speck_mc, 3 + 7 attack, k = 8, d = 2')
cal_DC(p0=2**(-6), p1=0.691816, p2=0.3963888, p3=0.2884552, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 7 attack, k = 16, d = 2
# dc = 2**20.215 = 1217012, t = 426505
print('deep_speck_mc, 3 + 7 attack, k = 16, d = 2')
cal_DC(p0=2**(-6), p1=0.6429904, p2=0.4566272, p3=0.3469392, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 7 attack, k = 2, d = 1
# dc = 2**20.349 = 1335212, t = 431589
print('deep_speck_mc, 3 + 7 attack, k = 2, d = 1')
cal_DC(p0=2**(-6), p1=0.599359, p2=0.4249366, p3=0.3199108, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 7 attack, k = 4, d = 1
# dc = 2**19.490 = 736302, t = 217069
print('deep_speck_mc, 3 + 7 attack, k = 4, d = 1')
cal_DC(p0=2**(-6), p1=0.6598372, p2=0.4311828, p3=0.290402, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 7 attack, k = 8, d = 1
# dc = 2**19.380 = 682043, t = 200073
print('deep_speck_mc, 3 + 7 attack, k = 8, d = 1')
cal_DC(p0=2**(-6), p1=0.691816, p2=0.4547288, p3=0.2884552, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 6 attack, k = 16, d = 1
# dc = 2**20.949 = 2024550, t = 710015
print('deep_speck_mc, 3 + 7 attack, k = 16, d = 1')
cal_DC(p0=2**(-6), p1=0.6429904, p2=0.4984928, p3=0.3469392, bp=0.005, bn=2**(-16))

print('')

# p0 = 2**(-12)
# deep_speck, 3 + 5 attack, d = 1
# dc = 2**26.657 = 105824624, t = 4909055
print('p0 = 2**(-12), deep_speck 3 + 5 attack, d = 1')
cal_DC(p0=2**(-12), p1=0.8976952, p2=0.3335031, p3=0.0462333, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 2, d = 1
# dc = 2**25.811 = 58866337, t = 1102668
print('p0=2**(-12), deep_speck_mc, 3 + 5 attack, k = 2, d = 1')
cal_DC(p0=2**(-12), p1=0.9665034, p2=0.4801842, p3=0.0185456, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 4, d = 1
# dc = 2**25.834 = 59834055, t = 426585
print('p0=2**(-12), deep_speck_mc, 3 + 5 attack, k = 4, d = 1')
cal_DC(p0=2**(-12), p1=0.9894372, p2=0.6927208, p3=0.0069172, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 8, d = 1
# dc = 2**24.888 = 31048883, t = 30816
print('p0=2**(-12), deep_speck_mc, 3 + 5 attack, k = 8, d = 1')
cal_DC(p0=2**(-12), p1=0.9987888, p2=0.8603952, p3=0.0007616, bp=0.005, bn=2**(-16))

# deep_speck_mc, 3 + 5 attack, k = 16, d = 1
# dc = 2**24.021 = 17020840, t = 4435
print('p0=2**(-12), deep_speck_mc, 3 + 5 attack, k = 16, d = 1')
cal_DC(p0=2**(-12), p1=0.9999072, p2=0.9672112, p3=1.92e-05, bp=0.005, bn=2**(-16))

print('')

# student distinguisher

# deep_speck_student, 3 + 6 attack, d = 2
# dc = 2**18.888 = 485068, t = 159043
print('student deep_speck, 3 + 6 attack, d = 2')
cal_DC(p0=2**(-6), p1=0.5969026, p2=0.371343, p3=0.3253696, bp=0.005, bn=2**(-8))

# deep_speck_mc_student, 3 + 6 attack, k = 2, d = 2
# dc = 2**17.785 = 225895, t = 67365
print('student deep_speck_mc, 3 + 6 attack, k = 2, d = 2')
cal_DC(p0=2**(-6), p1=0.6750112, p2=0.3533646, p3=0.2947486, bp=0.005, bn=2**(-8))
