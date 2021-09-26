import deep_net_mc as tn

tn.train_keccak_distinguisher(10, num_rounds=3, group_size=2, depth=1, diff=[(135, 0x80)])
