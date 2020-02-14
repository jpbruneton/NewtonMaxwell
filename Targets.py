#  ======================== MONTE CARLO TREE SEARCH ========================== #
# Project:          Symbolic regression tests
# Name:             EvaluateFit.py
# Description:      Tentative implementation of a basic MCTS
# Authors:          Vincent Reverdy & Jean-Philippe Bruneton
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import numpy as np
import Build_dictionnaries
import Simplification_rules
import config
from scipy import interpolate
from scipy.interpolate import griddata

# ============================================================================ #

class Target:

    def __init__(self, mode, fromfile = None):
        self.mode = mode  #'test' or 'train'
        self.from_file = fromfile

        if self.from_file is None: #define target with its analytical expression in target_list.txt
            self.target = self._define_target()

        else: #define target from data file
            if len(self.from_file) == 1:
                self.target = [self._definetargetfromfile()]
                #self.mytarget = 'dummytarget' #todo ct quoi ca??
            else:
                self.target = []
                for u in range(len(self.from_file)):
                    self.target.append(self._definetargetfromfile(u))

    def _definetargetfromfile(self, u=0):
        #adapter si plus de varables
        n_targets = 1
        n_variables = 1
        maximal_size = config.maxsize

        data = np.loadtxt(self.from_file[0], delimiter=',')

        x_train = data[:, 0]
        f0_train = data[:, 1]
        x_test = data[:, 0]
        f0_test = data[:, 1]
        print('taille target', self.mode, len(x_train))

        tck = interpolate.splrep(x_test, f0_test, s=0)
        yder_test = interpolate.splev(x_test, tck, der=1)
        ysec_test = interpolate.splev(x_test, tck, der=2)

        tck = interpolate.splrep(x_train, f0_train, s=0)
        yder_train = interpolate.splev(x_train, tck, der=1)
        ysec_train = interpolate.splev(x_train, tck, der=2)

        print('check important', x_train.size, f0_train.size, yder_train.size, ysec_train.size)
        f_normalization_train = np.amax(np.abs(f0_train))
        #f_normalization_train = 1/(0.0006103515625)**2
        #f0_train = f0_train / f_normalization_train

        f_normalization_train = 1

        #f_normalization_test = np.amax(np.abs(f0_test))
        #f_normalization_test = 1/(0.0006103515625)**2
        #f0_test = f0_test / f_normalization_test
        f_normalization_test = 1
        range_x_train = x_train[-1] - x_train[0]
        range_x_test = x_test[-1] - x_test[0]
        #range_x_train = 1
        #range_x_test = 1
        if self.mode == 'train':
            return n_targets, n_variables, [x_train], [np.asarray(f0_train)], f_normalization_train, [range_x_train], maximal_size, [yder_train, ysec_train]
        else:
            return n_targets, n_variables, [x_test], [np.asarray(f0_test)], f_normalization_test, [range_x_test], maximal_size, [yder_test, ysec_test]

    def _define_target(self):
        ''' Initialize game : builds the target given by target_list.txt '''
        # format of target_list.txt must be one target per line, and of the form:
        # n_targets, n_variables, expr1, train_set_type, train_set_range, test_set_type, test_set_range
        all_targets = []
        with open('target_list.txt') as myfile:
            for line in myfile:
                if line[0] != '#' and line[0] != '\n':
                    all_targets.append(line)

        print('all targets are: ', all_targets)
        to_return = []

        # ------ main loop on target list
        for elem in all_targets:
            mytarget = elem.replace(' ', '')
            mytarget = mytarget.replace('\n', '')
            mytarget = mytarget.split(',')
            name = mytarget[0]
            n_variables = int(mytarget[1])
            target_function = mytarget[2] #is a string, ok
            self.maximal_size = int(mytarget[-1])
            train_set_type_x = mytarget[3]
            test_set_type_x = mytarget[7]

            if train_set_type_x == 'E':
                train_set_range_x = [float(mytarget[4]), float(mytarget[5]), float(mytarget[6])]
                range_x_train = train_set_range_x[1] - train_set_range_x[0]
                x_train = np.linspace(train_set_range_x[0], train_set_range_x[1], num = int(range_x_train/train_set_range_x[2]))
            elif train_set_type_x == 'U':
                train_set_range_x = [float(mytarget[4]), float(mytarget[5]), int(mytarget[6])]
                range_x_train = train_set_range_x[1] - train_set_range_x[0]
                x_train = np.random.uniform(train_set_range_x[0], train_set_range_x[1], train_set_range_x[2])
                x_train = np.sort(x_train)
            else:
                print('training dataset not understood check spelling')
                raise ValueError

            if test_set_type_x == 'E':
                test_set_range_x = [float(mytarget[8]), float(mytarget[9]), float(mytarget[10])]
                range_x_test = test_set_range_x[1] - test_set_range_x[0]
                x_test = np.linspace(test_set_range_x[0], test_set_range_x[1], num=int((test_set_range_x[1]-test_set_range_x[0])/test_set_range_x[2]))
                print('taille train target', len(x_train))
            elif test_set_type_x == 'U':
                test_set_range_x = [float(mytarget[8]), float(mytarget[9]), int(mytarget[10])]
                range_x_test = test_set_range_x[1] - test_set_range_x[0]
                x_test = np.random.uniform(test_set_range_x[0], test_set_range_x[1], test_set_range_x[2])
                x_test = np.sort(x_test)
            else:
                print('testing dataset not understood')
                raise ValueError

            # ----------------------------------#
            if n_variables > 1:
                train_set_type_y = mytarget[11]
                test_set_type_y = mytarget[15]
                if train_set_type_y == 'E':
                    train_set_range_y = [float(mytarget[12]), float(mytarget[13]), float(mytarget[14])]
                    range_y_train = train_set_range_y[1] - train_set_range_y[0]
                    y_train = np.linspace(train_set_range_y[0], train_set_range_y[1], num=int(range_y_train/train_set_range_y[2]))
                elif train_set_type_y == 'U':
                    train_set_range_y = [float(mytarget[12]), float(mytarget[13]), int(mytarget[14])]
                    range_y_train = train_set_range_y[1] - train_set_range_y[0]
                    y_train = np.random.uniform(train_set_range_y[0], train_set_range_y[1], train_set_range_y[2])
                    y_train = np.sort(y_train)
                else:
                    print('training dataset not understood')
                    raise ValueError
                if test_set_type_y == 'E':
                    test_set_range_y = [float(mytarget[16]), float(mytarget[17]), float(mytarget[18])]
                    range_y_test = test_set_range_y[1] - test_set_range_y[0]
                    y_test = np.linspace(test_set_range_y[0], test_set_range_y[1], num=int((test_set_range_y[1]-test_set_range_y[0])/test_set_range_y[2]))
                elif test_set_type_y == 'U':
                    test_set_range_y = [float(mytarget[16]), float(mytarget[17]), int(mytarget[18])]
                    range_y_test = test_set_range_y[1] - test_set_range_y[0]
                    y_test = np.random.uniform(test_set_range_y[0], test_set_range_y[1], test_set_range_y[2])
                    y_test = np.sort(y_test)
                elif test_set_type_y == 'None':
                    y_test = y_train
                else:
                    print('testing dataset not understood')
                    raise ValueError

            if n_variables > 2:
                train_set_type_z = mytarget[19]
                test_set_type_z = mytarget[23]
                if train_set_type_z == 'E':
                    train_set_range_z = [float(mytarget[20]), float(mytarget[21]), float(mytarget[22])]
                    range_z_train = train_set_range_z[1] - train_set_range_z[0]
                    z_train = np.linspace(train_set_range_z[0], train_set_range_z[1], num=int(range_z_train/train_set_range_z[2]))
                elif train_set_type_z == 'U':
                    train_set_range_z = [float(mytarget[20]), float(mytarget[21]), int(mytarget[22])]
                    range_z_train = train_set_range_z[1] - train_set_range_z[0]
                    z_train = np.random.uniform(train_set_range_z[0], train_set_range_z[1], train_set_range_z[2])
                    z_train = np.sort(z_train)
                else:
                    print('training dataset not understood')
                    raise ValueError

                if test_set_type_z == 'E':
                    test_set_range_z = [float(mytarget[24]), float(mytarget[25]), float(mytarget[26])]
                    range_z_test = test_set_range_z[1] - test_set_range_z[0]
                    z_test = np.linspace(test_set_range_z[0], test_set_range_z[1], num=int((test_set_range_z[1]-test_set_range_z[0])/test_set_range_z[2]))
                elif test_set_type_z == 'U':
                    test_set_range_z = [float(mytarget[24]), float(mytarget[25]), int(mytarget[26])]
                    range_z_test = test_set_range_z[1] - test_set_range_z[0]
                    z_test = np.random.uniform(test_set_range_z[0], test_set_range_z[1], test_set_range_z[2])
                    z_test = np.sort(z_test)
                elif test_set_type_z == 'None':
                    z_test = z_train
                else:
                    print('testing dataset not understood')
                    raise ValueError

            if n_variables > 3:
                train_set_type_t = mytarget[27]
                test_set_type_t = mytarget[31]
                if train_set_type_t == 'E':
                    train_set_range_t = [float(mytarget[28]), float(mytarget[29]), float(mytarget[30])]
                    range_t_train = train_set_range_t[1] - train_set_range_t[0]
                    t_train = np.linspace(train_set_range_t[0], train_set_range_t[1], num=int(range_t_train/train_set_range_t[2]))
                elif train_set_type_t == 'U':
                    train_set_range_t = [float(mytarget[28]), float(mytarget[29]), int(mytarget[30])]
                    range_t_train = train_set_range_t[1] - train_set_range_t[0]
                    t_train = np.random.uniform(train_set_range_t[0], train_set_range_t[1], train_set_range_t[2])
                    t_train = np.sort(t_train)
                else:
                    print('training dataset not understood')
                    raise ValueError
                if test_set_type_t == 'E':
                    test_set_range_t = [float(mytarget[32]), float(mytarget[33]), float(mytarget[34])]
                    range_t_test = test_set_range_t[1] - test_set_range_t[0]
                    t_test = np.linspace(test_set_range_t[0], test_set_range_t[1], num=int((test_set_range_t[1]-test_set_range_t[0])/test_set_range_t[2]))
                elif test_set_type_t == 'U':
                    test_set_range_t = [float(mytarget[32]), float(mytarget[33]), int(mytarget[34])]
                    range_t_test = test_set_range_t[1] - test_set_range_t[0]
                    t_test = np.random.uniform(test_set_range_t[0], test_set_range_t[1], test_set_range_t[2])
                    t_test = np.sort(t_test)
                elif test_set_type_t == 'None':
                    t_test = t_train
                else:
                    print('testing dataset not understood')
                    raise ValueError

            # Then : ------------------------------------------------#
            if n_variables == 1:
                # eval functions
                x = x_train
                f0_train = eval(target_function)
                x = x_test
                f0_test = eval(target_function)
                # eval derivatives with interpolate
                tck = interpolate.splrep(x_train, f0_train, s=0)
                yder_train = interpolate.splev(x_train, tck, der=1)
                ysec_train = interpolate.splev(x_train, tck, der=2)
                tck = interpolate.splrep(x_test, f0_test, s=0)
                yder_test = interpolate.splev(x_test, tck, der=1)
                ysec_test = interpolate.splev(x_test, tck, der=2)
                range_x_train = 1
                range_x_test = 1
                if self.mode == 'train':
                    thistarget = [name, n_variables, [x_train / range_x_train], [f0_train], [range_x_train], self.maximal_size, [yder_train, ysec_train]]
                    to_return.append(thistarget)
                else:
                    thistarget = [name, n_variables, [x_test/range_x_test], [f0_test], [range_x_test], self.maximal_size, [yder_test, ysec_test]]
                    to_return.append(thistarget)

            elif n_variables == 2:
                # todo add interpolate deriv in higher dim??
                X_train, Y_train = np.meshgrid(x_train, y_train, indexing='ij')
                X_test, Y_test= np.meshgrid(x_test, y_test, indexing='ij')

                x, y = X_train, Y_train
                f0_train = eval(target_function)
                x, y = X_test, Y_test
                f0_test = eval(target_function)

                if self.mode == 'train':
                    thistarget = [name, n_variables, [X_train/range_x_train, Y_train/range_y_train], f0_train, [range_x_train, range_y_train], self.maximal_size]
                    to_return.append(thistarget)
                else:
                    thistarget = [name, n_variables, [X_test/range_x_test, Y_test/range_y_test], f0_test, [range_x_test, range_y_test], self.maximal_size]
                    to_return.append(thistarget)

            # ------------------------------------------------#
            elif n_variables == 3:
                X_train, Y_train, Z_train = np.meshgrid(x_train, y_train, z_train, indexing='ij')
                X_test, Y_test, Z_test = np.meshgrid(x_test, y_test, z_test, indexing='ij')

                x, y, z = X_train, Y_train, Z_train
                f0_train = eval(target_function)
                x, y, z = X_test, Y_test, Z_test
                f0_test = eval(target_function)

                if self.mode == 'train':
                    to_return = [name, n_variables, [X_train / range_x_train, Y_train / range_y_train, Z_train / range_z_train], f0_train, [range_x_train, range_y_train, range_z_train], self.maximal_size]
                    to_return.append(thistarget)
                else:
                    to_return = [name, n_variables, [X_test/range_x_test, Y_test/range_y_test, Z_test/range_z_test], f0_test, [range_x_test, range_y_test, range_z_test], self.maximal_size]
                    to_return.append(thistarget)

            elif n_variables == 4:

                X_train, Y_train, Z_train, T_train = np.meshgrid(x_train, y_train, z_train, t_train, indexing='ij')
                X_test, Y_test, Z_test, T_test = np.meshgrid(x_test, y_test, z_test, t_test, indexing='ij')

                x, y, z, t = X_train, Y_train, Z_train, T_train
                f0_train = eval(target_function)
                x, y, z, t = X_test, Y_test, Z_test, T_test
                f0_test = eval(target_function)

                print(f0_train.shape)
                # interpolate grid
                #grid_x, grid_y, grid_z, grid_t = np.mgrid[0:1:10j, 0:1:10j,0:1:10j,0:1:10j]
                #grid_f_train = griddata((x_train,y_train,z_train,t_train), f0_test, (grid_x, grid_y), method='linear')

                #mettons que cette data est E_x : on veut renvoyer les 4 derivées partielles d_x E_x
                # en soi de facon "exacte" ici d_i Ex c'est fo[i+1, :,:,:] _ f[i, :, :, :]
                first_derivative = []
                step_x = range_x_train/X_train.size
                first_derivative.append(np.diff(f0_train, axis=0)/step_x)
                step_y = range_y_train / Y_train.size
                first_derivative.append(np.diff(f0_train, axis=1) / step_y)
                step_z = range_z_train/Z_train.size
                first_derivative.append(np.diff(f0_train, axis=2)/step_z)
                step_t = range_t_train / T_train.size
                first_derivative.append(np.diff(f0_train, axis=3) / step_t)


                if self.mode == 'train':
                    thistarget = [name, n_variables, [X_train / range_x_train, Y_train / range_y_train, Z_train / range_z_train, T_train/range_t_train], f0_train, [range_x_train, range_y_train, range_z_train, range_t_train], self.maximal_size, first_derivative]
                    to_return.append(thistarget)
                else:
                    thistarget = [name, n_variables, [X_test / range_x_test, Y_test / range_y_test, Z_test / range_z_test, T_test/range_z_test], f0_test, [range_x_test,range_y_test,range_z_test, range_t_test], self.maximal_size, first_derivative]
                    to_return.append(thistarget)
        return to_return

class Voc():
    def __init__(self, target, modescalar):

        self.modescalar = modescalar
        if config.fromfile:
            self.target = target
        else:
            self.target = target
        if self.modescalar == 'noA':
            self.maximal_size = self.target[-2]
        else:
            self.maximal_size = self.target[-2]

        self.numbers_to_formula_dict, self.arity0symbols, self.arity1symbols, self.arity2symbols, self.true_zero_number, self.neutral_element, \
        self.infinite_number, self.terminalsymbol, self.OUTPUTDIM, self.pure_numbers, self.arity2symbols_no_power, self.power_number, \
        self.arity0symbols_var_and_tar, self.var_numbers, self.plusnumber, self.minusnumber, self.multnumber, self.divnumber, self.log_number, \
        self.exp_number, self.explognumbers, self.trignumbers, self.sin_number, self.cos_number \
            = Build_dictionnaries.get_dic(1, self.target[1], modescalar)
        self.outputdim = len(self.numbers_to_formula_dict) - 3

        self.mysimplificationrules, self.maxrulesize = self.create_dic_of_simplifs()

    def replacemotor(self, toreplace,replaceby, k):
        firstlist = []
        secondlist = []
        for elem in toreplace:
            if elem == 'zero':
                firstlist.append(self.true_zero_number)
            elif elem == 'neutral':
                firstlist.append(self.neutral_element)
            elif elem == 'infinite':
                firstlist.append(self.infinite_number)
            elif elem == 'scalar':
                firstlist.append(self.pure_numbers[0])
            elif elem == 'mult':
                firstlist.append(self.multnumber)
            elif elem == 'plus':
                firstlist.append(self.plusnumber)
            elif elem == 'minus':
                firstlist.append(self.minusnumber)
            elif elem == 'div':
                firstlist.append(self.divnumber)
            elif elem == 'variable':
                firstlist.append(self.var_numbers[k])
            elif elem == 'arity0':
                firstlist.append(self.arity0symbols[k])
            elif elem == 'fonction':
                firstlist.append(self.arity1symbols[k])
            elif elem == 'allops':
                firstlist.append(self.arity2symbols[k])
            elif elem == 'power':
                firstlist.append(self.power_number)
            elif elem == 'log':
                firstlist.append(self.log_number)
            elif elem == 'exp':
                firstlist.append(self.exp_number)
            elif elem == 'sin':
                firstlist.append(self.sin_number)
            elif elem == 'cos':
                firstlist.append(self.cos_number)
            elif elem == 'one':
                firstlist.append(self.pure_numbers[0])
            elif elem == 'two':
                firstlist.append(self.pure_numbers[1])
            else:
                print('bug1', elem)

        for elem in replaceby:
            if elem == 'zero':
                secondlist.append(self.true_zero_number)
            elif elem == 'neutral':
                secondlist.append(self.neutral_element)
            elif elem == 'infinite':
                secondlist.append(self.infinite_number)
            elif elem == 'scalar':
                secondlist.append(self.pure_numbers[0])
            elif elem == 'mult':
                secondlist.append(self.multnumber)
            elif elem == 'plus':
                secondlist.append(self.plusnumber)
            elif elem == 'minus':
                secondlist.append(self.minusnumber)
            elif elem == 'div':
                secondlist.append(self.divnumber)
            elif elem == 'variable':
                secondlist.append(self.var_numbers[k])
            elif elem == 'arity0':
                secondlist.append(self.arity0symbols[k])
            elif elem == 'fonction':
                secondlist.append(self.arity1symbols[k])
            elif elem == 'allops':
                secondlist.append(self.arity2symbols[k])
            elif elem == 'empty':
                secondlist=[]
            elif elem == 'power':
                secondlist.append(self.power_number)
            elif elem == 'log':
                secondlist.append(self.log_number)
            elif elem == 'exp':
                secondlist.append(self.exp_number)
            elif elem == 'sin':
                secondlist.append(self.sin_number)
            elif elem == 'cos':
                secondlist.append(self.cos_number)
            elif elem == 'one':
                secondlist.append(self.pure_numbers[0])
            elif elem == 'two':
                secondlist.append(self.pure_numbers[1])
            else:
                print('bug2', elem)

        return firstlist, secondlist



    def create_dic_of_simplifs(self):

        if self.modescalar == 'A':
            mydic_simplifs = {}
            for x in Simplification_rules.mysimplificationrules_with_A:
                toreplace = x[0]
                replaceby = x[1]

                if 'variable' in toreplace:
                    for k in range(self.target[1]):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'arity0' in toreplace:
                    for k in range(len(self.arity0symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'fonction' in toreplace:
                    for k in range(len(self.arity1symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'allops' in toreplace:
                    for k in range(len(self.arity2symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))
                else:
                    firstlist, secondlist = self.replacemotor(toreplace, replaceby, 0)
                    mydic_simplifs.update(({str(firstlist): secondlist}))

            maxrulesize = 0
            for i in range(len(Simplification_rules.mysimplificationrules_with_A)):
                if len(Simplification_rules.mysimplificationrules_with_A[i][0]) > maxrulesize:
                    maxrulesize = len(Simplification_rules.mysimplificationrules_with_A[i][0])

            return mydic_simplifs, maxrulesize

        if self.modescalar == 'noA':
            mydic_simplifs = {}
            for x in Simplification_rules.mysimplificationrules_no_A:
                toreplace = x[0]
                replaceby = x[1]
                if 'variable' in toreplace:
                    for k in range(self.target[1]):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'arity0' in toreplace:
                    for k in range(len(self.arity0symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'fonction' in toreplace:
                    for k in range(len(self.arity1symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                elif 'allops' in toreplace:
                    for k in range(len(self.arity2symbols)):
                        firstlist, secondlist = self.replacemotor(toreplace, replaceby, k)
                        mydic_simplifs.update(({str(firstlist): secondlist}))

                else:
                    firstlist, secondlist = self.replacemotor(toreplace, replaceby, 0)
                    mydic_simplifs.update(({str(firstlist): secondlist}))

            maxrulesize = 0

            for i in range(len(Simplification_rules.mysimplificationrules_no_A)):
                if len(Simplification_rules.mysimplificationrules_no_A[i][0]) > maxrulesize:
                    maxrulesize = len(Simplification_rules.mysimplificationrules_no_A[i][0])

            return mydic_simplifs, maxrulesize