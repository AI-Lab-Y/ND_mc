import deep_net_mc_gohr as tn

# during the training of ND7, Gohr set depth=1
# when num_rounds=7, group_size = 4, 8, 16,
# we can not obtain a valid distinguisher by using the network proposed by Gohr even if we let depth = 10
for group_size in [4, 8, 16]:
    tn.train_speck_distinguisher(10, num_rounds=7, diff=(0x40, 0), depth=1, group_size=group_size)






