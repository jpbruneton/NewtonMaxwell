import config
from copy import deepcopy

def get_dic(n_targets, all_targets_name, u, calculus_mode, look_for, expert_knowledge):
    explicit_time_dependence, no_first_derivatives, use_distance, planar_motion = expert_knowledge

    # ============  arity 0 symbols =======================
    if calculus_mode == 'scalar':
        my_dic_scalar_number = ['A']
        my_dic_vec_number = []
    else:
        my_dic_scalar_number = ['A'] #we need both e.g. F = q v^B in Lorentz force : q is a one_scalar
        my_dic_vec_number = ['B']

    my_dic_special_scalar = []
    if use_distance:
        if planar_motion:
            my_dic_special_scalar=['f0**2+f1**2']
        else:
            my_dic_special_scalar = ['f0**2+f1**2+f3**2']
    # targets :
    my_dic_other_targets = all_targets_name[:u] + all_targets_name[u+1:]
    my_dic_actual_target = [all_targets_name[u]]

    # variable is only time here:
    if explicit_time_dependence:
        my_dic_variables = ['x0']
    else:
        my_dic_variables = []

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
    if no_first_derivatives:
        my_dic_diff = []
    # ============  arity 1 symbols =======================
    # basic functions
    my_dic_scalar_functions = config.fonctions
    # not clear if I allow point wise operations of cos( vec(x)) as hadamard like structure or use only on true scalars?
    # decide later : for now default is cos can only apply on a true scalar like :  A_vec (scalarproduct) F_2
    if calculus_mode == 'vectorial':
        my_dic_vectorial_functions = ['la.norm('] # forms from E -> R
    else:
        my_dic_vectorial_functions = []

    # ============  arity 2 symbols =======================
    my_dic_scalar_operators = config.operators
    my_dic_power = ['**']

    if calculus_mode == 'vectorial':
        my_dic_vec_operators = ['np.vdot(', 'np.cross('] #operators from E*E -> R
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
    arity0dic = my_dic_scalar_number + my_dic_vec_number + my_dic_special_scalar+ my_dic_variables + my_dic_actual_target\
                + my_dic_other_targets + my_dic_diff
    index0 = 2
    index1 = index0 + len(my_dic_scalar_number) + len(my_dic_vec_number)+len(my_dic_special_scalar)
    index2 = index1 + len(my_dic_variables)

    target_function_number =2 + len(my_dic_scalar_number) + len(my_dic_vec_number)+len(my_dic_special_scalar) + len(my_dic_variables)
    first_der_number = 2 + len(my_dic_scalar_number) + len(my_dic_vec_number)+len(my_dic_special_scalar) + len(my_dic_variables) + len(my_dic_actual_target)+len(my_dic_other_targets)
    # todo ci dessus c'est un tuple rangé si multitarget


    if calculus_mode == 'vectorial' and len(my_dic_vec_number) !=0:
        vectorial_numbers = [index0 +  len(my_dic_scalar_number)]
    if calculus_mode == 'vectorial':
        vectorial_numbers.extend([i for i in range(index2, 2 + len(arity0dic))])
        arity0_vec = tuple(deepcopy(vectorial_numbers))
    if explicit_time_dependence:
        arity0_novec = (index0, index0+1+len(my_dic_vec_number)+len(my_dic_special_scalar))
    else:
        arity0_novec = [index0]

    pure_numbers = tuple([i for i in range(index0, index1)])
    var_numbers = tuple([i for i in range(index1, index2)])

    #arity 1 and 2 :
    arity1dic = my_dic_scalar_functions + my_dic_vectorial_functions
    arity2dic = my_dic_scalar_operators + my_dic_vec_operators + my_dic_power
    a0, a1, a2  = len(arity0dic), len(arity1dic), len(arity2dic)
    arity_1_novec = [i for i in range(2 + a0, 2 + a0 + len(my_dic_scalar_functions))]

    if calculus_mode == 'vectorial':
        vectorial_numbers.extend([i for i in range(2+a0+len(my_dic_scalar_functions), 2+a0+a1)])
        arity1_vec = 2+a0+len(my_dic_scalar_functions)
        vectorial_numbers.extend([i for i in range(2+a0+a1+len(my_dic_scalar_operators), 2+a0+a1+len(my_dic_scalar_operators) + len(my_dic_vec_operators))])
        arity2_vec = tuple([i for i in range(2+a0+a1+len(my_dic_scalar_operators), 2+a0+a1+len(my_dic_scalar_operators) + len(my_dic_vec_operators))])

    if calculus_mode == 'vectorial':
        vectorial_numbers = tuple(vectorial_numbers)
        print(vectorial_numbers)
    else:
        vectorial_numbers = None
        arity0_vec = None
        arity1_vec = None
        arity2_vec = None
    arity0symbols = tuple([i for i in range(2, 2 + a0)])
    arity1symbols = tuple([i for i in range(2 + a0, 2 + a0 + a1)])
    arity2symbols = tuple([i for i in range(2 + a0 + a1, 2 + a0 + a1 + a2)])
    arity2symbols_no_power = tuple([i for i in range(2 + a0 + a1, 2 + a0 + a1 + a2 -1)])
    if calculus_mode == 'vectorial':
        arity2symbols_novec = tuple([x for x in arity2symbols if x not in arity2_vec])
    else:
        arity2symbols_novec = arity2symbols
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

    all =[numbers_to_formula_dict, arity0symbols, arity1symbols, arity2symbols, true_zero_number, neutral_element, \
           infinite_number, terminalsymbol, pure_numbers, arity2symbols_no_power, power_number, var_numbers, \
           plusnumber, minusnumber, multnumber, divnumber, norm_number, dotnumber, wedgenumber, vectorial_numbers, \
           arity0_vec, arity0_novec, arity1_vec, arity2_vec, arity2symbols_novec, arity_1_novec, target_function_number, first_der_number]
    for elem in all:
        print(elem)

    {'1': 'halt', '2': 'A', '3': 'B', '4': 'F0', '5': 'np.sin(', '6': 'np.sqrt(', '7': 'np.exp(', '8': 'np.log(',
     '9': 'la.norm(', '10': '+', '11': '-', '12': '*', '13': '/', '14': 'np.vdot(', '15': 'np.cross(', '16': '**',
     '17': 'zero', '18': 'neutral', '19': 'infinity'}

    return numbers_to_formula_dict, arity0symbols, arity1symbols, arity2symbols, true_zero_number, neutral_element, \
           infinite_number, terminalsymbol, pure_numbers, arity2symbols_no_power, power_number, var_numbers, \
           plusnumber, minusnumber, multnumber, divnumber, norm_number, dotnumber, wedgenumber, vectorial_numbers, \
           arity0_vec, arity0_novec, arity1_vec, arity2_vec, arity2symbols_novec, arity_1_novec, target_function_number, first_der_number

