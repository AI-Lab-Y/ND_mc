import deep_net as dn

dn.train_des_distinguisher(10, num_rounds=6, diff=(0x40080000, 0x04000000), depth=1)

