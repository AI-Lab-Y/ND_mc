import deep_net_mc_jk_rk as dn
import des_jk_rk as des


n = 10**7
n_eval = 10**6
diff = (0x40080000, 0x04000000)
for nr in [5, 6]:
    for group_size in [2, 4, 8, 16]:
        X_j, X, Y = des.make_dataset(n=n, nr=nr, group_size=group_size, diff=diff)
        X_j_eval, X_eval, Y_eval = des.make_dataset(n=n_eval, nr=nr, group_size=group_size, diff=diff)
        dn.train_joint_key_distinguisher(10, X_j, Y, X_j_eval, Y_eval, num_rounds=nr, group_size=group_size, depth=1)
        dn.train_random_key_distinguisher(10, X, Y, X_eval, Y_eval, num_rounds=nr, group_size=group_size, depth=1)
