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

#import matplotlib.pyplot as plt

import sys
# ============================ CLASS: Evaluate_Fit ================================ #
# A class returning the reward of a given equation w.r.t. the target data (or function)
class Evaluatefit:

    def __init__(self, formulas, voc, target, mode, formal_target):
        self.formulas = copy.deepcopy(formulas)
        self.target = target
        self.voc = voc
        self.mode = mode
        self.scalar_numbers = 0
        if formal_target == 'der_sec':
            self.maxder = 2
        else:
            self.maxder = 1
        self.name, self.n_variables, self.variables, self.targets, \
        self.ranges, self.maximal_size, self.derivatives = self.target

        self.xsize = self.variables[0].shape[0]
        self.stepx = (self.ranges[0])/self.xsize #only works if mode = 'E'

        if self.n_variables > 1:
            self.ysize = self.variables[0].shape[1]
            self.stepy = (self.ranges[1]) / self.ysize  # only works if mode = 'E'
        if self.n_variables > 2:
            self.zsize = self.variables[0].shape[2]
            self.stepz = (self.ranges[2]) / self.zsize  # only works if mode = 'E'
        if self.n_variables >3:
            self.tsize = self.variables[0].shape[3]
            self.stept = (self.ranges[3]) / self.tsize  # only works if mode = 'E'

        if config.madifftarget == 0:
            self.mavraitarget = self.targets[0]
        if config.madifftarget == 'dx':
            self.mavraitarget = self.derivatives[0]
        elif config.madifftarget == 'dy':
            self.mavraitarget = self.derivatives[1]
        elif config.madifftarget == 'dz':
            self.mavraitarget = self.derivatives[2]
        elif config.madifftarget == 'dt':
            self.mavraitarget = self.derivatives[3]

    # ---------------------------------------------------------------------------- #
    def rename_formulas(self):
        ''' index all the scalar 'A' by a A1, A2, etc, rename properly the differentials, and finally resize as it must '''
        neweq = ''

        # rename the A's
        self.scalar_numbers = self.formulas.count('A')

        if self.scalar_numbers == 1:
            self.formulas += '+ A'
            self.scalar_numbers = 2

        A_count = 0
        for char in self.formulas:
            if char == 'A':
                neweq += 'A[' + str(A_count) + ']'
                A_count += 1
            else:
                neweq += char
        print(neweq)
        highest_der = 0
        for u in range(1,self.maxder):
            if 'd'*u in neweq:
                highest_der += 1

        if highest_der != 0 :

            for u in range(1, highest_der+1):
                if highest_der-u > 0:
                    arr = '[:-' + str(highest_der-u) + ']'
                    arr = '[:]'
                else:
                    arr = '[:]'
                look_for = 'd'*u + '_x0'*u + '_f0'
                replace_by = 'np.diff(f[0]' + arr + ',' +str(u) +')/('  + str(self.step) +'**' +str(u)+')'
                replace_by = 'f'+'p'*u
                neweq = neweq.replace(look_for, replace_by)

        if highest_der != 0 :
            base_array = '[:-' + str(highest_der) + ']'
            base_array = '[:]'
        else:
            base_array = '[:]'
        string_to_replace = 'x0'
        replace_by = '(x[0]' + base_array +')'#+ '*' + str(self.ranges[0]) + ')'
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace = 'f0'
        replace_by = 'f[0]' + base_array
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace  = 'one'
        replace_by = '1.0'
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace = 'two'
        replace_by = '2.0'
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace = 'neutral'
        replace_by = '1.0'
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace = 'zero'
        replace_by = '0.0'
        neweq = neweq.replace(string_to_replace, replace_by)
        self.formulas = neweq

    # ---------------------------------------------------------------------------- #
    def formula_eval(self, x, f, fp, A) :
        try:
            mafonction = eval(self.formulas)
            if type(mafonction) != np.ndarray or np.isnan(np.sum(mafonction)) or np.isinf(np.sum(mafonction)) :
                return False, None
            else:
                monx = x[0]
                if config.monobj ==0:
                    toreturn = mafonction
                if config.monobj ==1 :
                    tck = interpolate.splrep(monx, mafonction, s=0)
                    mader = interpolate.splev(monx, tck, der=1)
                    toreturn = mader
                    if self.formulas == 'np.sin((x[0][:]))' and False:
                        plt.plot(mafonction, 'r')
                        plt.plot(mader, 'b')
                        #tets = np.sin(monx)
                        #plt.plot(tets, 'g')
                        #tt = np.cos(monx)
                        #plt.plot(tt)
                        plt.show()

                else:
                    tck = interpolate.splrep(monx, mafonction, s=0)
                    mader = interpolate.splev(monx, tck, der=1)
                    madersec = interpolate.splev(monx, tck, der=2)
                return True, toreturn

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):

            return False, None

    # ---------------------------------------------------------------------------- #
    def evaluation_target(self, a):
        err = 0
        success, eval = self.formula_eval(self.variables, self.targets, self.derivatives[0], a)

        if success == True:
            resize_eval = eval[:self.mavraitarget.size]
            diff = resize_eval - self.mavraitarget
            err += (np.sum(diff**2))
            err /= np.sum(np.ones_like(diff))
        else:
            return 1200000

        return err

    # -----------------------------------------
    def finish_with_least_squares_target(self, a):
        # this flattens the training data : this is required for least squares method : must be a size-1 array!
        flatfun = []
        res = self.func(a, self.targets, self.variables)

        if self.n_variables == 1:
            for i in range(self.xsize):
                flatfun.append(res[i])

        if self.n_variables == 2:
            for i in range(self.xsize):
                for j in range(self.ysize):
                    flatfun.append(res[i, j])

        if self.n_variables == 3:
            for i in range(self.xsize):
                for j in range(self.ysize):
                    for k in range(self.zsize):
                        flatfun.append(res[i, j, k])

        return np.asarray(flatfun)

    # ---------------------------------------------------------------------------- #
    def fit(self, reco):
        # applies least square fit starting from the recommendation of cmaes :
        x0 = reco
        return least_squares(self.finish_with_least_squares_target, x0, jac='2-point', loss='cauchy', args=())

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
    def best_A_least_squares(self, reco):
    # calls least square fit from cmaes reco :
        try:
            ls_attempt = self.fit(reco)

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):
            return False, [1]*self.scalar_numbers

        success = ls_attempt.success
        if success:
            reco_ls = ls_attempt.x
            # transforms array into list
            rec = []
            for u in range(reco_ls.size):
                rec.append(reco_ls[u])
            return True, rec

        else:
            return False, [1]*self.scalar_numbers


    def eval_reward_nrmse(self, A):
    #for validation only
        success, result = self.formula_eval(self.variables, self.targets, self.derivatives[0], A)
        if success:
            resize_result = result[:self.mavraitarget.size]
            quadratic_cost = np.sum((self.mavraitarget - resize_result)**2)
            n = self.mavraitarget.size
            rmse = np.sqrt(quadratic_cost / n)
            nrmse = rmse / np.std(self.mavraitarget)
            return nrmse

        else:
            return 100000000

    # ------------------------------------------------------------------------------- #
    def evaluate(self):
        ''' evaluate the reward of an equation'''

        np.seterr(all = 'ignore')
        allA = []
        failure_reward = -1
        self.rename_formulas()
        if self.scalar_numbers == 0:
            rms = self.eval_reward_nrmse(allA)
            if rms > 100000000:
                rms = 100000000
            return self.scalar_numbers, allA, rms

        # else: cmaes fit : ---------------------------------------------------------- #
        success, allA = self.best_A_cmaes()
        if success == False:
            return self.scalar_numbers, [1]*self.scalar_numbers, 100000000

        # ---------------------------------------------------------------------------- #
        #else, compute some actual reward:
        rms = self.eval_reward_nrmse(allA)

        #now compare the three and chose the best
        if rms > 100000000:
            rms = 100000000

        return self.scalar_numbers, allA, rms