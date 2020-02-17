import config

def get_dic(n_targets, all_targets_name, u, calculus_mode, look_for):

    # ============  arity 0 symbols =======================
    if calculus_mode == 'only_scalars':
        my_dic_scalar_numbers = ['A_scal']
        my_dic_vec_numbers = []
    elif calculus_mode== 'both':
        my_dic_scalar_numbers = ['A_scal']
        my_dic_vec_numbers = []
    else:
        my_dic_scalar_numbers = []
        my_dic_vec_numbers = ['A_vec']

    # targets :
    my_dic_other_targets = all_targets_name[:u] + all_targets_name[u+1:]
    my_dic_actual_target = all_targets_name[u]
    my_dic_target_vector = []
    for i in range(n_targets//3):
        my_dic_target_vector.append('F'+str(i)) # F0 = vec(f0,f1,f2), etc ; #assumes 3D dynamics

    # variable is only time here:
    my_dic_variables = ['x0']

    #derivatives #are considered as scalars here, not via a node operator (#later todo?)
    if look_for == 'find_1st_order_diff_eq':
        maxder = 1
    elif look_for == 'find_2nd_order_diff_eq':
        maxder=2
    else:
        maxder = 0

    my_dic_scalar_diff = []
    my_dic_vector_diff = []
    ders = [['']]
    while len(ders) < maxder:
        loc_der = []
        pre_der = ders[-1]
        for elem in pre_der:
                loc_der.append('d' + elem + '_x0')
        ders.append(loc_der)

    for u in range(n_targets):
        for i in range(1, len(ders)):
            for der in ders[i]:
                my_dic_scalar_diff.append(der + '_f' + str(u))

    for u in range(len(my_dic_target_vector)):
        for i in range(1, len(ders)):
            for der in ders[i]:
                my_dic_vector_diff.append(der + '_F' + str(u))


    # ============  arity 1 symbols =======================
    # basic functions
    my_dic_scalar_functions = config.fonctions
    my_dic_vec_functions = ['norm(', 'normsquared(']

    # ============  arity 2 symbols =======================
    my_dic_scalar_operators = config.operators
    if config.usepower:
        my_dic_power = ['**']
    else:
        my_dic_power = []
    my_dic_vec_operators = ['dot', 'wedge']

    # ============ special algebraic symbols =======================
    my_dic_true_zero = ['zero']
    my_dic_neutral = ['neutral']
    my_dic_infinite =['infinity']
    special_dic = my_dic_true_zero + my_dic_neutral + my_dic_infinite

    # --------------------------
    #concatenate the dics
    numbers_to_formula_dict = {'1' : 'halt'}

    arity0dic = my_dic_any_scalars + my_dic_variables + my_dic_diff + my_dic_targets
    pure_numbers = tuple([i for i in range(2, 2 + len(my_dic_any_scalars))])
    var_numbers = tuple([i for i in range(2 + len(my_dic_any_scalars), 2 + len(my_dic_any_scalars) + len(my_dic_variables))])


    #then :
    arity1dic = my_dic_functions
    #in both cases:
    arity2dic = my_dic_regular_op + my_dic_power

    a0 = len(arity0dic)
    a1 = len(arity1dic)
    a2 = len(arity2dic)



    arity0symbols_var_and_tar = tuple([i for i in range(2 + len(my_dic_any_scalars), 2 + a0)])



    arity0symbols = tuple([i for i in range(2, 2 + a0)])
    arity1symbols = tuple([i for i in range(2 + a0, 2 + a0 + a1)])

    arity2symbols = tuple([i for i in range(2 + a0 + a1, 2 + a0 + a1 + a2)])
    if config.usepower:
        arity2symbols_no_power = tuple([i for i in range(2 + a0 + a1, 2 + a0 + a1 + a2 -1)])
    else:
        arity2symbols_no_power = tuple([i for i in range(2 + a0 + a1, 2 + a0 + a1 + a2)])

    #dont change order of operations!
    plusnumber = 2 + a0 + a1
    minusnumber = 3 + a0 + a1
    multnumber = 4 + a0 + a1
    divnumber = 5 + a0 + a1

    #or the order of special dic
    if config.usepower:
        power_number = 1 + a0 + a1 + a2
    else:
        power_number = None

    true_zero_number = 2 + a0 + a1 + a2
    neutral_element = 3 + a0 + a1 + a2
    infinite_number = 4 + a0 + a1 + a2

    # finally
    all_my_dics = arity0dic + arity1dic + arity2dic + special_dic
    # and
    for i in range(len(all_my_dics)):
        numbers_to_formula_dict.update({str(i+2): all_my_dics[i]})

    terminalsymbol = 1

    #for Neural Net
    OUTPUTDIM = len(numbers_to_formula_dict)

    #check everything's fine
    if False:
        print(modescalar)
        print(arity0symbols)
        print(arity1symbols)
        print(arity2symbols_no_power)
        print(arity2symbols)
        print(pure_numbers)
        print(power_number)
        print(true_zero_number)
        print(neutral_element)
        print(infinite_number)
        print(arity0symbols_var_and_tar)
        print(log_number)
        print(exp_number)
        print(plusnumber)
        print(minusnumber)
        print(multnumber)
        print(divnumber)
        print(explognumbers)
        print(trignumbers)
        print('========================')

    return numbers_to_formula_dict, arity0symbols, arity1symbols, arity2symbols, true_zero_number, neutral_element, \
           infinite_number, terminalsymbol, OUTPUTDIM, pure_numbers, arity2symbols_no_power, power_number,  \
           arity0symbols_var_and_tar, var_numbers, plusnumber, minusnumber, multnumber, divnumber, log_number, exp_number, \
           explognumbers, trignumbers, sin_number, cos_number

