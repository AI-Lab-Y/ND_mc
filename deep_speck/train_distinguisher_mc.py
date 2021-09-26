import deep_net_mc as tn

tn.train_speck_distinguisher(20, num_rounds=5, diff=(0x40, 0), group_size=2, depth=1)

# tn.train_speck_distinguisher(10, num_rounds=5, diff=(0x40, 0), group_size=4, depth=1)
#
# tn.train_speck_distinguisher(10, num_rounds=5, diff=(0x40, 0), group_size=8, depth=1)
#
# tn.train_speck_distinguisher(10, num_rounds=5, diff=(0x40, 0), group_size=16, depth=1)
