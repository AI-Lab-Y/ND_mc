import deep_net_mc_jk_rk as dn
import chaskey_jk_rk as chk


n = 10**7
n_eval = 10**6
nr = 6
group_size = 4
diff = (0x8400, 0x0400, 0, 0)
x = 4
y = 0
head = 0
X_j, X, Y = chk.make_train_data(n=n, x=x, y=y, head=head, group_size=group_size, diff=group_size)
X_j_eval, X_eval, Y_eval = chk.make_train_data(n=n, x=x, y=y, head=head, group_size=group_size, diff=group_size)
dn.train_joint_key_distinguisher(10, X_j, Y, X_j_eval, Y_eval, num_rounds=nr, group_size=group_size, depth=1)
dn.train_random_key_distinguisher(10, X, Y, X_eval, Y_eval, num_rounds=nr, group_size=group_size, depth=1)
