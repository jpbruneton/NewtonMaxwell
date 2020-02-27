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
import matplotlib.pyplot as plt
import config
import copy
import cma
from scipy import interpolate
from numpy import linalg as la

#import matplotlib.pyplot as plt

import sys
# ============================ CLASS: Evaluate_Fit ================================ #
# A class returning the reward of a given equation w.r.t. the target data (or function)
class Evaluatefit:

    def __init__(self, formulas, voc, train_targets, mode, u, look_for):
        self.calculus_mode = voc.calculus_mode
        self.formulas = copy.deepcopy(formulas)
        self.train_targets = train_targets
        self.u = u
        self.look_for = look_for
        self.voc = voc
        self.mode = mode
        self.scalar_numbers = 0
        self.maximal_size = voc.maximal_size

        self.variable = [self.train_targets[0][1]] #00 is the name
        self.mytarget = self.train_targets[u]
        self.mytarget_function = self.mytarget[2]
        self.mytarget_1der = self.mytarget[3]
        self.mytarget_2der = self.mytarget[4]
        self.array_functions = [x[2] for x in self.train_targets]
        self.array_first_der = [x[3] for x in self.train_targets]
        self.size = self.variable[0].size
        if look_for == 'find_2nd_order_diff_eq':
            self.maxder = 2
            self.objectivefunction = self.mytarget_2der
        elif look_for == 'find_1st_order_diff_eq':
            self.maxder = 1
            self.objectivefunction = self.mytarget_1der
        elif look_for == 'find_function':
            self.maxder = 0
            self.objectivefunction = self.mytarget_function

    # ---------------------------------------------------------------------------- #
    def rename_formulas(self):
        ''' index all the scalar 'A' by a A1, A2, etc, rename properly the differentials, and finally resize as it must '''

        #print('entering')
        #print(self.formulas)
        self.scalar_numbers = self.formulas.count('A')
        self.v_numbers = self.formulas.count('B')

        if self.calculus_mode == 'scalar':
            if self.scalar_numbers == 1: #cmaes must be at least 2 dim
                self.formulas += '+ A'
                self.scalar_numbers = 2

            neweq = ''
            A_count = 0
            for char in self.formulas:
                if char == 'A':
                    neweq += 'S[' + str(A_count) + ']'
                    A_count += 1
                else:
                    neweq += char
        else:
            if self.scalar_numbers == 1 and self.v_numbers ==0: #cmaes must be at least 2 dim
                self.formulas += '+ B'
                self.v_numbers+=1

            #first rename B to an array of scalars:
            string_to_replace = 'B'
            replace_by = 'np.array([A*ones,A*ones,A*ones])'
            self.formulas = self.formulas.replace(string_to_replace, replace_by)
            count = 0

            neweq=''
            for char in self.formulas:
                if char == 'A':
                    neweq += 'S[' + str(count) + ']'
                    count += 1
                else:
                    neweq+=char

        highest_der = 0
        for u in range(1,self.maxder):
            if 'd'*u in neweq:
                highest_der += 1

        if highest_der != 0 :
            for u in range(1, highest_der+1):
                for v in range(len(self.train_targets)):
                    if self.calculus_mode == 'scalar':
                        toreplace = 'd'*u + '_x0'*u + '_f'+str(v)
                        replace_by = 'f' + 'p' * u + str(v)

                    else:
                        toreplace = 'd'*u + '_x0'*u + '_F'+str(v)
                        replace_by = 'F' + 'p' * u + str(v)

                    neweq = neweq.replace(toreplace, replace_by)

        string_to_replace = 'x0'
        replace_by = '(x[0][:])'
        neweq = neweq.replace(string_to_replace, replace_by)

        for u in range(len(self.train_targets)):
            if self.calculus_mode == 'scalar':
                string_to_replace = 'f'+str(u)
                replace_by = 'f['+str(u)+'][:]'
            else:
                string_to_replace = 'F' + str(u)
                replace_by = 'F[' + str(u) + ']'

            neweq = neweq.replace(string_to_replace, replace_by)

            if self.calculus_mode == 'scalar':
                string_to_replace = 'fp' + str(u)
                replace_by = 'fp[' + str(u) + '][:]'
            else:
                string_to_replace = 'Fp' + str(u)
                replace_by = 'Fp[' + str(u) + ']'
            neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace = 'SIZE'
        replace_by = str(self.size)
        neweq = neweq.replace(string_to_replace, replace_by)

        self.formulas = neweq

        print('rename', neweq)
    # ---------------------------------------------------------------------------- #
    def formula_eval(self, x, f, fp, S) :
        try:
            mafonction = eval(self.formulas)
            if type(mafonction) != np.ndarray or np.isnan(np.sum(mafonction)) or np.isinf(np.sum(mafonction)) :
                return False, None
            else:
                toreturn = mafonction
                return True, toreturn

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):
            return False, None

    # ------------------
    def formula_eval_vectorial(self, x, F, Fp, S) :
        ones = np.ones(x[0].size)
        try:
            mafonction = eval(self.formulas)
            if type(mafonction) != np.ndarray or np.isnan(np.sum(mafonction)) or np.isinf(np.sum(mafonction)) :
                return False, None
            else:
                toreturn = mafonction
                return True, toreturn

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):
            return False, None

    # ---------------------------------------------------------------------------- #
    def evaluation_target_vectorial(self, S):
        err = 0
        success, eval = self.formula_eval_vectorial(self.variable, self.array_functions, self.array_first_der, S)

        if success == True:
            diff = eval - self.objectivefunction
            err += (np.sum(diff ** 2))
            err /= np.sum(np.ones_like(diff))
        else:
            return 1200000

        return err

    # ---------------------------------------------------------------------------- #
    def evaluation_target(self, a):
        err = 0
        success, eval = self.formula_eval(self.variable, self.array_functions, self.array_first_der, a)

        if success == True:
            diff = eval - self.objectivefunction
            err += (np.sum(diff**2))
            err /= np.sum(np.ones_like(diff))
        else:
            return 1200000

        return err

    # ---------------------------------------------------------------------------- #
    def func(self, A, f, x):
        # eval the func, leaves only A undefined
        toeval = self.formulas + '-f'
        return eval(toeval)

    # -------------------------------------------------------------------------------  #
    def best_A_cmaes(self):
        # applies the cmaes fit:
        initialguess = 2*np.random.rand(self.scalar_numbers)-1
        initialsigma = np.random.randint(1,5)

        try:
            res = cma.CMAEvolutionStrategy(initialguess, initialsigma,
                {'verb_disp': 0}).optimize(self.evaluation_target).result

            reco = res.xfavorite
            rec = []

            for u in range(reco.size):
                rec.append(reco[u])

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):

            return False, [1]*self.scalar_numbers

        return True, rec

    # -------------------------------------------------------------------------------  #
    def best_vectorial_cmaes(self):
        # applies the cmaes fit:
        initialguess = 2 * np.random.rand(self.scalar_numbers+3*self.v_numbers) - 1
        initialsigma = np.random.randint(1, 5)
        try:
            res = cma.CMAEvolutionStrategy(initialguess, initialsigma,
                                           {'verb_disp': 0}).optimize(self.evaluation_target_vectorial).result
            reco = res.xfavorite
            rec = []
            for u in range(reco.size):
                rec.append(reco[u])

        except (
        RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):
            return False, [1] * self.scalar_numbers
        return True, rec
    # ------------------------------------------------------------------------------- #
    def eval_reward_nrmse(self, S):
    #for validation only
        success, result = self.formula_eval(self.variable, self.array_functions, self.array_first_der, S)
        if success:
            quadratic_cost = np.sum((self.objectivefunction - result)**2)
            n = self.objectivefunction.size

            rmse = np.sqrt(quadratic_cost / n)
            nrmse = rmse / np.std(self.objectivefunction)
            return nrmse

        else:
            return 100000000

    # ------------------------------------------------------------------------------- #
    def eval_reward_nrmse_vectorial(self, S):
    #for validation only
        success, result = self.formula_eval_vectorial(self.variable, self.array_functions, self.array_first_der, S)
        if success:
            quadratic_cost = np.sum((self.objectivefunction - result)**2)
            n = self.objectivefunction.size
            rmse = np.sqrt(quadratic_cost / n)
            nrmse = rmse / np.std(self.objectivefunction)
            return nrmse

        else:
            return 100000000

    # ------------------------------------------------------------------------------- #
    def evaluate(self):
        ''' evaluate the reward of an equation'''

        np.seterr(all = 'ignore')

        if self.calculus_mode == 'scalar':
            allS = []
            self.rename_formulas()
            if self.scalar_numbers == 0:
                rms = self.eval_reward_nrmse(allS)
                if rms > 100000000:
                    rms = 100000000
                return self.scalar_numbers, allS, rms

            # else: cmaes fit : ---------------------------------------------------------- #
            success, allS = self.best_A_cmaes()
            if success == False:
                return self.scalar_numbers, [1] * self.scalar_numbers, 100000000

            # ---------------------------------------------------------------------------- #
            # else, compute some actual reward:
            rms = self.eval_reward_nrmse(allS)

            # now compare the three and chose the best
            if rms > 100000000:
                rms = 100000000

            return self.scalar_numbers, allS, rms

        else:
            if self.formulas == 'B':
                print('ici')
            self.rename_formulas()
            allS = []
            if self.scalar_numbers == 0 and self.v_numbers == 0:
                rms = self.eval_reward_nrmse_vectorial(allS)
                if rms > 100000000:
                    rms = 100000000
                return self.scalar_numbers, allS, rms

            # else: cmaes fit : ---------------------------------------------------------- #
            success, allS = self.best_vectorial_cmaes()
            if success == False:
                return self.scalar_numbers + 3*self.v_numbers, [1] * (self.scalar_numbers+3*self.v_numbers), 100000000

            # ---------------------------------------------------------------------------- #
            # else, compute some actual reward:
            rms = self.eval_reward_nrmse_vectorial(allS)

            # now compare the three and chose the best
            if rms > 100000000:
                rms = 100000000

            return (self.scalar_numbers + 3*self.v_numbers), allS, rms

