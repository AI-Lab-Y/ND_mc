import deep_net_mc as dn

# dn.train_des_distinguisher(10, num_rounds=6, diff=(0x40080000, 0x04000000), group_size=2, depth=1)
# dn.train_des_distinguisher(10, num_rounds=6, diff=(0x40080000, 0x04000000), group_size=4, depth=1)
dn.train_des_distinguisher(10, num_rounds=6, diff=(0x40080000, 0x04000000), group_size=8, depth=1)
# dn.train_des_distinguisher(10, num_rounds=6, diff=(0x40080000, 0x04000000), group_size=16, depth=1)

