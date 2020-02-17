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

# ============================================================================ #

class Target():

    def __init__(self, filenames):
        self.filenames = filenames
        self.targets = []
        for u in range(len(filenames)):
            self.targets.append(self._definetargetfromfile(u))

    def _definetargetfromfile(self, u):

        data = np.loadtxt(self.filenames[u], delimiter=',')
        n_targets = data.shape[1] -1

        t = data[:, 0]
        f0 = data[:, 1]
        target_functions = [f0]

        tck = interpolate.splrep(t, f0, s=0)
        f0der = interpolate.splev(t, tck, der=1)
        f0sec = interpolate.splev(t, tck, der=2)

        first_derivatives = [f0der]
        second_derivatives = [f0sec]

        if n_targets >1:
            f1 = data[:, 2]
            target_functions.append(f1)
            tck = interpolate.splrep(t, f1, s=0)
            f1der = interpolate.splev(t, tck, der=1)
            f1sec = interpolate.splev(t, tck, der=2)
            first_derivatives.append(f1der)
            second_derivatives.append(f1sec)

        if n_targets>2:
            f2 = data[:, 3]
            target_functions.append(f2)
            tck = interpolate.splrep(t, f2, s=0)
            f2der = interpolate.splev(t, tck, der=1)
            f2sec = interpolate.splev(t, tck, der=2)
            first_derivatives.append(f2der)
            second_derivatives.append(f2sec)

        if n_targets > 3:
            print('not supported yet; vectors should be no more than in 3D space')
            raise ValueError

        return [t, target_functions, first_derivatives, second_derivatives]



class Voc():
    def __init__(self, u, all_targets_name, calculus_mode, maximal_size, look_for):
        self.calculus_mode = calculus_mode
        self.maximal_size = maximal_size
        self.all_targets_name = all_targets_name
        self.n_targets = len(all_targets_name)
        self.look_for = look_for
        self.numbers_to_formula_dict, self.arity0symbols, self.arity1symbols, self.arity2symbols, self.true_zero_number, self.neutral_element, \
        self.infinite_number, self.terminalsymbol, self.OUTPUTDIM, self.pure_numbers, self.arity2symbols_no_power, self.power_number, \
        self.arity0symbols_var_and_tar, self.var_numbers, self.plusnumber, self.minusnumber, self.multnumber, self.divnumber, self.log_number, \
        self.exp_number, self.explognumbers, self.trignumbers, self.sin_number, self.cos_number \
            = Build_dictionnaries.get_dic(self.n_targets, self.all_targets_name, u, self.calculus_mode, self.look_for)

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