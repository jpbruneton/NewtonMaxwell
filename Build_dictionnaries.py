import config

def get_dic(n_targets, all_targets_name, u, calculus_mode, look_for):

    # ============  arity 0 symbols =======================
    if calculus_mode == 'scalar':
        my_dic_scalar_number = ['A']
        my_dic_vec_number = []
    else:
        my_dic_scalar_number = ['A_scal'] #we need both e.g. F = q v^B in Lorentz force : q is a one_scalar
        my_dic_vec_number = ['A_vec']

    # targets :
    my_dic_other_targets = all_targets_name[:u] + all_targets_name[u+1:]
    my_dic_actual_target = [all_targets_name[u]]

    # variable is only time here:
    my_dic_variables = ['x0']

    #derivatives #are considered as scalars here, not via a node operator (#later todo?)
    if look_for == 'find_1st_order_diff_eq':
        maxder = 1
    elif look_for == 'find_2nd_order_diff_eq':
        maxder=2
    else:
        maxder = 0

    my_dic_diff = []
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
                if calculus_mode == 'scalar':
                    my_dic_diff.append(der + '_f' + str(u))
                else:
                    my_dic_diff.append(der + '_F' + str(u))

    # ============  arity 1 symbols =======================
    # basic functions
    my_dic_scalar_functions = config.fonctions
    # not clear if I allow point wise operations of cos( vec(x)) as hadamard like structure or use only on true scalars?
    # decide later : for now default is cos can only apply on a true scalar like :  A_vec (scalarproduct) F_2
    if calculus_mode == 'vectorial':
        my_dic_vectorial_functions = ['norm('] # forms from E -> R
    else:
        my_dic_vectorial_functions = []

    # ============  arity 2 symbols =======================
    my_dic_scalar_operators = config.operators
    my_dic_power = ['**']

    if calculus_mode == 'vectorial':
        my_dic_vec_operators = ['dot', 'wedge'] #operators from E*E -> R
    else:
        my_dic_vec_operators = []

    # ============ special algebraic symbols =======================
    my_dic_true_zero = ['zero']
    my_dic_neutral = ['neutral']
    my_dic_infinite =['infinity']
    special_dic = my_dic_true_zero + my_dic_neutral + my_dic_infinite

    # --------------------------
    #concatenate the dics
    numbers_to_formula_dict = {'1' : 'halt'}

    #arity 0:
    arity0dic = my_dic_scalar_number + my_dic_vec_number + my_dic_variables + my_dic_actual_target\
                + my_dic_other_targets + my_dic_diff
    index0 = 2
    index1 = index0 + len(my_dic_scalar_number) + len(my_dic_vec_number)
    index2 = index1 + len(my_dic_variables)

    pure_numbers = tuple([i for i in range(index0, index1)])
    var_numbers = tuple([i for i in range(index1, index2)])

    #arity 1 and 2 :
    arity1dic = my_dic_scalar_functions + my_dic_vectorial_functions
    arity2dic = my_dic_scalar_operators + my_dic_vec_operators + my_dic_power

    a0, a1, a2  = len(arity0dic), len(arity1dic), len(arity2dic)

    arity0symbols = tuple([i for i in range(2, 2 + a0)])
    arity1symbols = tuple([i for i in range(2 + a0, 2 + a0 + a1)])
    arity2symbols = tuple([i for i in range(2 + a0 + a1, 2 + a0 + a1 + a2)])
    arity2symbols_no_power = tuple([i for i in range(2 + a0 + a1, 2 + a0 + a1 + a2 -1)])

    norm_number = 2+a0+a1-1
    #dont change order of operations!
    plusnumber = 2 + a0 + a1
    minusnumber = 3 + a0 + a1
    multnumber = 4 + a0 + a1
    divnumber = 5 + a0 + a1
    if calculus_mode == 'vectorial':
        dotnumber = 6+a0+a1
        wedgenumber = 7+a0+a1
    else:
        dotnumber = None
        wedgenumber = None

    #dont change the previous order or change this accordingly
    power_number = 1 + a0 + a1 + a2
    true_zero_number = 2 + a0 + a1 + a2
    neutral_element = 3 + a0 + a1 + a2
    infinite_number = 4 + a0 + a1 + a2

    # finally
    all_my_dics = arity0dic + arity1dic + arity2dic + special_dic
    for i in range(len(all_my_dics)):
        numbers_to_formula_dict.update({str(i+2): all_my_dics[i]})

    terminalsymbol = 1


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
           infinite_number, terminalsymbol, pure_numbers, arity2symbols_no_power, power_number, var_numbers, \
           plusnumber, minusnumber, multnumber, divnumber, norm_number, dotnumber, wedgenumber

