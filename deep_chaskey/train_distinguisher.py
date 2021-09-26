import deep_net as tn

tn.train_chaskey_distinguisher(10, num_rounds=3, diff=(0x8400, 0x0400, 0, 0),  depth=1)

