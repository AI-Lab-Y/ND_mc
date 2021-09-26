import numpy as np

bits_num = 16

def hw(v):
  res = np.zeros(v.shape, dtype=np.uint8)
  for i in range(16):
    res = res + ((v >> i) & 1)

  return(res)

low_weight = np.array(range(2**bits_num), dtype=np.uint16)
low_weight = hw(low_weight)
# print('low weight is ', low_weight)
# num = [1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1]
num = [np.sum(low_weight == i) for i in range(bits_num+1)]
low_weight_8_bits = np.array(range(2**8), dtype=np.uint8)
low_weight_8_bits = hw(low_weight_8_bits)
num_8_bits = [np.sum(low_weight_8_bits == i) for i in range(8+1)]


def show_attack_res_with_reduced_key_space(th1, th2, cnt1_path, cnt2_path, tk_path, time_cost_path):
    cnt1 = np.load(cnt1_path)
    cnt2 = np.load(cnt2_path)
    true_key = np.load(tk_path).astype(np.uint16)
    time_cost = np.load(time_cost_path)
    attack_num = len(true_key)
    print("Attack number is", attack_num)
    print('Average attack time cost is {}s'.format(np.mean(time_cost)))
    surviving_num_divided_by_d_stage_1 = np.zeros((attack_num, 8+1), dtype=np.uint32)
    surviving_num_divided_by_d_stage_2 = np.zeros((attack_num, 16+1), dtype=np.uint32)
    surviving_num_stage_1 = np.sum(cnt1 > th1, axis=1)
    surviving_num_stage_2 = np.sum(cnt2 > th2, axis=1)
    average_cnt_divided_by_d_stage_1 = np.zeros(8+1)
    average_cnt_divided_by_d_stage_2 = np.zeros(16+1)
    effect_number_divided_by_d_stage_2 = np.zeros(16+1, dtype=np.uint32)
    # gen surviving_num_divided_by_d_stage_1
    for i in range(attack_num):
        for j in range(2**8):
            dp = (true_key[i] & 0xff) ^ np.uint8(j)
            dp = low_weight[dp]
            average_cnt_divided_by_d_stage_1[dp] += cnt1[i][j]
            if cnt1[i][j] > th1:
                surviving_num_divided_by_d_stage_1[i][dp] += 1
    # gen surviving_num_divided_by_d_stage_2
    for i in range(attack_num):
        for j in range(2**16):
            dp = true_key[i] ^ np.uint16(j)
            dp = low_weight[dp]
            if cnt2[i][j] > 0:
                average_cnt_divided_by_d_stage_2[dp] += cnt2[i][j]
                effect_number_divided_by_d_stage_2[dp] += 1
            if cnt2[i][j] > th2:
                surviving_num_divided_by_d_stage_2[i][dp] += 1
    for i in range(8+1):
        average_cnt_divided_by_d_stage_1[i] /= (attack_num * num_8_bits[i])
    for i in range(16+1):
        if effect_number_divided_by_d_stage_2[i] == 0:
            effect_number_divided_by_d_stage_2[i] = 1
    average_cnt_divided_by_d_stage_2 = average_cnt_divided_by_d_stage_2 / effect_number_divided_by_d_stage_2

    # show index_a
    print("The number that the right key(d=0) passes the statistical test in stage1 is", np.sum(surviving_num_divided_by_d_stage_1[:,0]))
    print("The number that the right key(d=0) passes the statistical test in stage2 is", np.sum(surviving_num_divided_by_d_stage_2[:,0]))

    # show index_b
    print("The average number of surviving keys in stage1 in {} trails is {}".format(attack_num, np.sum(surviving_num_divided_by_d_stage_1) / attack_num))
    print("The average number of surviving keys in stage2 in {} trails is {}".format(attack_num, np.sum(surviving_num_divided_by_d_stage_2) / attack_num))

    # show index_c
    surviving_num_upper_bound_stage_1 = 37.67
    surviving_num_upper_bound_stage_2 = 137.31
    print("The number that the number of surviving keys is smaller than {} in stage1 is {}".format(surviving_num_upper_bound_stage_1, np.sum(surviving_num_stage_1<surviving_num_upper_bound_stage_1)))
    print("The number that the number of surviving keys is smaller than {} in stage2 is {}".format(surviving_num_upper_bound_stage_2, np.sum(surviving_num_stage_2<surviving_num_upper_bound_stage_2)))

    # show surviving number and average cnt according to d
    # stage1
    average_surviving_rate_divided_by_d_stage_1 = np.mean(surviving_num_divided_by_d_stage_1, axis=0) / num_8_bits
    for i in range(8+1):
        print("d =", i, end=', ')
        # average count
        print("The average count in stage1 is {:.3f}".format(average_cnt_divided_by_d_stage_1[i]), end=', ')
        # average pass rate
        print("The average pass rate in stage1 is {:.3f}".format(average_surviving_rate_divided_by_d_stage_1[i]))
    # stage2
    average_surviving_rate_divided_by_d_stage_2 = np.mean(surviving_num_divided_by_d_stage_2, axis=0) / num
    for i in range(16+1):
        print("d =", i, end=', ')
        # average count
        print("The average count in stage2 is {:.3f}".format(average_cnt_divided_by_d_stage_2[i]), end=', ')
        # average pass rate
        print("The average pass rate in stage2 is {:.3f}".format(average_surviving_rate_divided_by_d_stage_2[i]))

# deep_speck
print('attack result for deep_speck:')
prepend_rounds = 3
dis_nr = 6
c2 = 0.5
n1 = 485068
th1 = 159043
n2 = 123209
th2 = 19147
folder = './key_recovery_record/{}_{}_{}_{}_{}_{}_{}/'.format(prepend_rounds, dis_nr, c2, n1, th1, n2, th2)
cnt1_path = folder + 'cnt_1.npy'
cnt2_path = folder + 'cnt_2.npy'
tk_path = folder + 'true_key.npy'
time_cost_path = folder + 'time_consumption.npy'
show_attack_res_with_reduced_key_space(th1=th1, th2=th2, cnt1_path=cnt1_path, cnt2_path=cnt2_path, tk_path=tk_path, time_cost_path=time_cost_path)

print('')

# deep_speck_mc
print('attack result for deep_speck_mc:')
prepend_rounds = 3
dis_nr = 6
c2 = 0.5
n1 = 225895
th1 = 67365
n2 = 57895
th2 = 6104
folder = './key_recovery_record/{}_{}_{}_{}_{}_{}_{}/'.format(prepend_rounds, dis_nr, c2, n1, th1, n2, th2)
cnt1_path = folder + 'cnt_1.npy'
cnt2_path = folder + 'cnt_2.npy'
tk_path = folder + 'true_key.npy'
time_cost_path = folder + 'time_consumption.npy'
show_attack_res_with_reduced_key_space(th1=th1, th2=th2, cnt1_path=cnt1_path, cnt2_path=cnt2_path, tk_path=tk_path, time_cost_path=time_cost_path)