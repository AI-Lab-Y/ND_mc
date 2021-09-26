import test_key_recovery as gohr
import test_key_recovery_mc as mc
import numpy as np
from keras.models import model_from_json,load_model


#generate test data
def gen_data(n):
    ct_all=[]
    ct_mc_all=[]
    key_all=[]
    for j in range(n):
        ct, ct_mc, key = mc.gen_challenge(n=100, nr=11, neutral_bits=[20,21,22,14,15,23], keyschedule='real')
        ct_all.append(ct)
        ct_mc_all.append(ct_mc)
        key_all.append(key)
    return ct_all,ct_mc_all,key_all

# load distinguishers
mc_net7 = load_model('./saved_model/7_2_mc_distinguisher.h5')
mc_net6 = load_model('./saved_model/6_2_mc_distinguisher.h5')

json_file = open('./saved_model/single_block_resnet.json','r')
json_model = json_file.read()
gohr_net7 = model_from_json(json_model)
gohr_net6 = model_from_json(json_model)
gohr_net7.load_weights('./saved_model/net7_small.h5')
gohr_net6.load_weights('./saved_model/net6_small.h5')
gohr_net6_10 = load_model('./saved_model/6_depth_10_distinguisher.h5')


for i in range(10):
    # expriment for 100 times
    n = 100
    ct_all,ct_mc_all,key_all = gen_data(n)
    arr1_mc, arr2_mc, good_mc, tt_mc=mc.test(index=i,n=n,ct_all=ct_mc_all,key_all=key_all,num_structures=100,cutoff1=18,cutoff2=150,net=mc_net7,net_help=mc_net6)
    np.save(open('./key_recovery_record/mc/res'+ str(i) + 'run_sols1.npy','wb'),arr1_mc)
    np.save(open('./key_recovery_record/mc/res'+ str(i) + 'run_sols2.npy','wb'),arr2_mc)
    np.save(open('./key_recovery_record/mc/res'+ str(i) + 'run_good.npy','wb'),good_mc)
    np.save(open('./key_recovery_record/mc/res'+ str(i) + 'run_time.npy','wb'),tt_mc)
    # recommended cutoff parameter of gohr is (10,10)
    # arr1, arr2, good=gohr.test(index=i,n=n,ct_all=ct_all,key_all=key_all,num_structures=100,cutoff1=10,cutoff2=10,net=gohr_net7,net_help=gohr_net6_10)
    arr1, arr2, good=gohr.test(index=i,n=n,ct_all=ct_all,key_all=key_all,num_structures=100,cutoff1=10,cutoff2=10,net=gohr_net7,net_help=gohr_net6)
    np.save(open('./key_recovery_record/gohr/res'+ str(i) + 'run_sols1.npy','wb'),arr1)
    np.save(open('./key_recovery_record/gohr/res'+ str(i) + 'run_sols2.npy','wb'),arr2)
    np.save(open('./key_recovery_record/gohr/res'+ str(i) + 'run_good.npy','wb'),good)