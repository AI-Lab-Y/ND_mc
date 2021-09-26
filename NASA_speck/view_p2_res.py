import numpy as np

# if you want to see p1, p2, p3 together, open txt file. p3 isn't included in following npy file

def view_selected_file(path='./'):
    res = np.load(path)
    print(res)

# deep_speck_teacher, nr = 5, c = 0.5
nr = 5
c = 0.5
print('deep_speck_teacher, nr = {}, c = {}:'.format(nr, c))
p2_d1_path = './p2_estimation_res/teacher/{}/{}_{}_p2_d1.npy'.format(nr, nr, c)
view_selected_file(p2_d1_path)

# deep_speck_teacher, nr = 7, c = 0.5
nr = 6
c = 0.5
print('deep_speck_teacher, nr = {}, c = {}:'.format(nr, c))
p2_d1_path = './p2_estimation_res/teacher/{}/{}_{}_p2_d1.npy'.format(nr, nr, c)
view_selected_file(p2_d1_path)

# deep_speck_teacher, nr = 7, c = 0.5
nr = 7
c = 0.5
print('deep_speck_teacher, nr = {}, c = {}:'.format(nr, c))
p2_d1_path = './p2_estimation_res/teacher/{}/{}_{}_p2_d1.npy'.format(nr, nr, c)
view_selected_file(p2_d1_path)

# deep_speck_student, nr = 6, c = 0.5
nr = 6
c = 0.5
print('deep_speck_student, nr = {}, c = {}:'.format(nr, c))
p2_d1_path = './p2_estimation_res/student/{}/{}_{}_p2_d1.npy'.format(nr, nr, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 5, k = 2, c = 0.5
nr = 5
k = 2
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 5, k = 4, c = 0.5
nr = 5
k = 4
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 5, k = 8, c = 0.5
nr = 5
k = 8
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 5, k = 16, c = 0.5
nr = 5
k = 16
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 6, k = 2, c = 0.5
nr = 6
k = 2
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 6, k = 4, c = 0.5
nr = 6
k = 4
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)
# deep_speck_mc_teacher, nr = 6, k = 8, c = 0.5
nr = 6
k = 8
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 6, k = 16, c = 0.5
nr = 6
k = 16
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 7, k = 2, c = 0.5
nr = 7
k = 2
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 7, k = 4, c = 0.5
nr = 7
k = 4
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 7, k = 8, c = 0.5
nr = 7
k = 8
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_teacher, nr = 7, k = 16, c = 0.5
nr = 7
k = 16
c = 0.5
print('deep_speck_mc_teacher, nr = {}, k = {}, c = {}:'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)

# deep_speck_mc_student, nr = 6, k = 2, c = 0.5
nr = 6
k = 2
c = 0.5
print('deep_speck_mc_student, nr = {}, k = {}, c = {}'.format(nr, k, c))
p2_d1_path = './p2_estimation_res/mc_student/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, k, c)
view_selected_file(p2_d1_path)