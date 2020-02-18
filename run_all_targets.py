import run_one_target
import config
from Targets import Target, Voc
import numpy as np

# -----------------------------------------------#
def init_parameters(actual_train_target, all_targets_name, look_for, calculus_mode, maximal_size, u, expert_knowledge):
    # init dictionnaries
    voc = Voc(u, all_targets_name, calculus_mode, maximal_size, look_for, expert_knowledge)
    print('for target name', actual_train_target[0])
    print('we work with voc: ', voc.numbers_to_formula_dict)

    # and metaparameters
    poolsize = config.qd_init_pool_size
    if config.MAX_DEPTH ==1:
        delete_ar1_ratio = 0.3
    elif config.MAX_DEPTH == 2:
        delete_ar1_ratio = 0.3
    else:
        delete_ar1_ratio = 0.8
    extend_ratio = config.extendpoolfactor
    p_mutate = 0.4
    p_cross = 0.8

    binl_no_a = maximal_size # number of bins for length of an eq
    maxl_no_a = maximal_size
    bina = maxl_no_a  # number of bins for number of free scalars
    maxa = bina
    binl_a = maximal_size # number of bins for length of an eq
    maxl_a = maximal_size
    binf = 160 # number of bins for number of fonctions
    maxf = 160
    new = 0
    binp = new  # number of bins for number of powers
    maxp = new
    derzero, derone = 1 , 1 #absence ou presence de fo et ou de fo'
    addrandom = config.add_random

    params = [poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa, binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
           binp, maxp, derzero, derone, addrandom, voc]
    return params

# -----------------------------------------------#
def kill_print():
    import sys
    class writer(object):
        log = []

        def write(self, data):
            self.log.append(data)

    logger = writer()
    sys.stdout = logger
    sys.stderr = logger

# -----------------------------------------------#
def load_targets(filenames_train, filenames_test, flatten, expert_knowledge):
    train = Target(filenames_train).targets
    test = Target(filenames_test).targets
    alltraintargets = []
    alltesttargets = []
    count = 0
    planar_motion = expert_knowledge[3]
    eps = 1 if planar_motion else 0
    if flatten:
        for u in range(len(train)):
            for v in range(len(train[u][1])-eps):
                name = 'f'+str(count)
                onetarget = [name, train[u][0], train[u][1][v],train[u][2][v],train[u][3][v]]
                count+=1
                alltraintargets.append(onetarget)
        for u in range(len(test)):
            for v in range(len(test[u][1])-eps):
                name = 'f'+str(count)
                onetarget = [name, test[u][0], test[u][1][v], test[u][2][v], test[u][3][v]]
                count+=1
                alltesttargets.append(onetarget)
        return alltraintargets, alltesttargets

    else:
        for u in range(len(train)):
            name = 'F'+str(u)
            onetarget =  [name, train[u][0], train[u][1],train[u][2],train[u][3]]
            alltraintargets.append(onetarget)
            onetarget = [name, test[u][0], test[u][1], test[u][2], test[u][3]]
            alltesttargets.append(onetarget)

        return alltraintargets, alltesttargets

# ----------------------------------------------
def init_targets(calculus_mode, calculus_modes, expert_knowledge):
    # check if possible
    error = False
    for file in filenames_train:
        dat = np.loadtxt(file, delimiter=',')
        if dat.shape[1] != 4:
            error = True
    if error:
        print('warning: vector mode can only be used if *all* targets are 3D; now resuming with scalar mode only')
        calculus_mode = calculus_modes[0]

    # we dont load the target the same in these two cases:
    # if flatten == True, train of the form: [[name, variable array, one scalar target, its derivative, its second derivative] , n times]
    # else, [[name, var array, vec target, vec derivatives], p times] ; with 3*p = n
    flatten = True if calculus_mode == 'scalar' else False
    train_targets, test_targets = load_targets(filenames_train, filenames_test, flatten, expert_knowledge)

    all_targets_name = [train_targets[u][0] for u in range(len(train_targets))]
    return  calculus_mode, all_targets_name, train_targets, test_targets

# -----------------------------------------------#
if __name__ == '__main__':
    # don't display any output
    noprint = False
    if noprint:
        kill_print()

    filenames_train = ['data_loader/kepler_1.csv']#,'data_loader/x2_train(t).csv']
    filenames_test = ['data_loader/kepler_1.csv']#,'data_loader/x2_test(t).csv']

    #allow expert knowledge :
    explicit_time_dependence = True
    no_first_derivatives = False
    use_distance = False
    planar_motion = True
    expert_knowledge =[explicit_time_dependence, no_first_derivatives, use_distance, planar_motion]

    # -------------------------------- init targets
    calculus_modes = ['scalar', 'vectorial']
    calculus_mode = calculus_modes[1] #default is vectorial mode on
    calculus_mode, all_targets_name, train_targets, test_targets = init_targets(calculus_mode, calculus_modes, expert_knowledge)

    # solve :
    for u in range(len(train_targets)):
        actual_train_target = train_targets[u]
        actual_test_target = test_targets[u]
        possible_modes = ['find_function', 'find_1st_order_diff_eq', 'find_2nd_order_diff_eq', 'find_primitive']
        look_for = possible_modes[2] #defaults is second order diff eq
        maximal_size = 20
        # main exec
        params = init_parameters(actual_train_target, all_targets_name, look_for, calculus_mode, maximal_size, u, expert_knowledge)
        run_one_target.main(params, train_targets, test_targets, u, look_for, calculus_mode, maximal_size)
