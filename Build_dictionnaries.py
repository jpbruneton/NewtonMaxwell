import config

def get_dic(n_targets, n_variables, modescalar):
    # =================   stop symbol
    my_dic_halt=['halt']

    # ============  arity 0 symbols =======================
    my_dic_any_scalars = ['A']
    my_dic_integer_scalars = config.list_scalars
    # targets are f0, f1, ...

    if config.includetarget:
        my_dic_targets = []
        if n_targets == 0:
            print('there must be at least one target function')
            raise ValueError
        else:
            for i in range(n_targets):
                my_dic_targets.append('f' + str(i))
    else:
        my_dic_targets = []

    # variables are x1, x2, ...
    my_dic_variables = []
    for i in range(n_variables):
        my_dic_variables.append('x' + str(i))

    # partial derivatives w.r.t. x1, x2, ...
    # see also game_env : d_i are only allowed acting directly on f_j, ie no stuff like partial_x (x^2+ cos(y) - f) : because could always be simplified and thus useless
    # todo later : change to arity two avec seul x1, x2 etc allowed
    if config.use_derivative:
        my_dic_diff = []
        ders = [['']]
        if n_variables == 1:
            maxder = 2
        else:
            maxder = 1
        while len(ders) < maxder:
            loc_der = []
            pre_der = ders[-1]
            for elem in pre_der:
                for v in range(n_variables):
                    loc_der.append('d' + elem + '_x' + str(v))
            ders.append(loc_der)

        for u in range(n_targets):
            for i in range(1, len(ders)):
                for der in ders[i]:
                    my_dic_diff.append(der + '_f' + str(u))
    else:
        my_dic_diff = []

    # ============  arity 1 symbols =======================

    # basic functions
    my_dic_functions = config.fonctions


    # ============  arity 2 symbols =======================
    my_dic_regular_op = config.operators
    if config.usepower:
        my_dic_power = ['**']
    else:
        my_dic_power = []

    # ============ special algebraic symbols =======================
    my_dic_true_zero = ['zero']
    my_dic_neutral = ['neutral']
    my_dic_infinite =['infinity']
    special_dic = my_dic_true_zero + my_dic_neutral + my_dic_infinite

    # --------------------------
    #concatenate the dics
    numbers_to_formula_dict = {'1' : 'halt'}

    if modescalar == 'A':
        arity0dic = my_dic_any_scalars + my_dic_variables + my_dic_diff + my_dic_targets
        pure_numbers = tuple([i for i in range(2, 2 + len(my_dic_any_scalars))])
        var_numbers = tuple([i for i in range(2 + len(my_dic_any_scalars), 2 + len(my_dic_any_scalars) + len(my_dic_variables))])

    else:
        arity0dic = my_dic_integer_scalars + my_dic_variables + my_dic_diff + my_dic_targets
        pure_numbers = tuple([i for i in range(2, 2 + len(my_dic_integer_scalars))])
        var_numbers = tuple([i for i in range(2 + len(my_dic_integer_scalars), 2 + len(my_dic_integer_scalars) + len(my_dic_variables))])

    #then :
    arity1dic = my_dic_functions
    #in both cases:
    arity2dic = my_dic_regular_op + my_dic_power

    a0 = len(arity0dic)
    a1 = len(arity1dic)
    a2 = len(arity2dic)

    #later remove too bog grid sinon
    if 'np.log(' in my_dic_functions:
        log_number = 2 + a0 + [x for x in range(len(my_dic_functions)) if my_dic_functions[x]=='np.log('][0]
    else:
        log_number = None
    if 'np.exp(' in my_dic_functions:
        exp_number = 2 + a0 + [x for x in range(len(my_dic_functions)) if my_dic_functions[x]=='np.exp('][0]
    else:
        exp_number = None
    explognumbers = (exp_number, log_number)
    if 'np.sin(' in my_dic_functions:
        sin_number = 2 + a0 + [x for x in range(len(my_dic_functions)) if my_dic_functions[x] == 'np.sin('][0]
    else:
        sin_number = None
    if 'np.cos(' in my_dic_functions:
        cos_number = 2 + a0 + [x for x in range(len(my_dic_functions)) if my_dic_functions[x] == 'np.cos('][0]
    else:
        cos_number = None
    trignumbers = (sin_number, cos_number)


    if modescalar == 'A':
        arity0symbols_var_and_tar = tuple([i for i in range(2 + len(my_dic_any_scalars), 2 + a0)])
    else:
        arity0symbols_var_and_tar = tuple([i for i in range(2 + len(my_dic_integer_scalars), 2 + a0)])

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

