import deep_net_mc_jk_rk as dn
import speck_jk_rk as sp


n = 10**7
n_eval = 10**6
nr = 6
group_size = 4
diff = (0x0040, 0)
X_j, X, Y = sp.make_train_data(n=n, nr=nr, group_size=group_size, diff=diff)
X_j_eval, X_eval, Y_eval = sp.make_train_data(n=n_eval, nr=nr, group_size=group_size, diff=diff)
dn.train_joint_key_distinguisher(10, X_j, Y, X_j_eval, Y_eval, num_rounds=nr, group_size=group_size, depth=1)
dn.train_random_key_distinguisher(10, X, Y, X_eval, Y_eval, num_rounds=nr, group_size=group_size, depth=1)
