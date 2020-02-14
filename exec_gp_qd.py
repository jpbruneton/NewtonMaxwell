import os
from gp_qd_class import printresults, GP_QD
import pickle
import multiprocessing as mp
import config
from game_env import Game
import game_env
from Targets import Target, Voc
from State import State
import time
import csv
import sys
import numpy as np
from Evaluate_fit import Evaluatefit
import pickle
# -------------------------------------------------------------------------- #
def init_grid(reinit_grid, poolname):

    if config.uselocal == False:
        filepath = '/home/user/results/'+poolname
        with open(filepath, 'rb') as file:
            qdpool = pickle.load(file)
            file.close()
            print('loading already trained model')
            print('with', len(qdpool))
    else:
        if reinit_grid:
            if os.path.exists(poolname):
                os.remove(poolname)

        if os.path.exists(poolname) and config.saveqd:
            print('loading already trained model')

            with open(poolname, 'rb') as file:
                qdpool = pickle.load(file)
                file.close()

        else:
            print('grid doesnt exist')
            qdpool = None

    #time.sleep(1)

    return qdpool

############
def save_qd_pool(pool, type):
    timeur = int(time.time()*1000000)
    if config.uselocal:
        file_path = 'QD_pool' + type + '.txt'
    else:
        file_path = '/home/user/results/QD_pool' + type + str(timeur) + '.txt'

    with open(file_path, 'wb') as file:
        pickle.dump(pool, file)
        file.close()

# -------------------------------------------------------------------------- #
def evalme(onestate):
    test_target, voc, state, u = onestate[0], onestate[1], onestate[2], onestate[3]

    results = []
    scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, voc, test_target, 'test',u)
    results.append([rms, scalar_numbers, alla])

    if config.tworunsineval and voc.modescalar == 'A':
        # run 2:
        scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, voc, test_target, 'test',u)
        results.append([rms, scalar_numbers, alla])

        if results[0][0] <= results[1][0]:
            rms, scalar_numbers, alla = results[0]
        else:
            rms, scalar_numbers, alla = results[1]

    if state.reversepolish[-1] == voc.terminalsymbol:
        L = len(state.reversepolish) -1
    else:
        L = len(state.reversepolish)

    if voc.modescalar == 'noA':
        scalar_numbers =0
        for char in voc.pure_numbers :
            scalar_numbers += state.reversepolish.count(char)

        scalar_numbers += state.reversepolish.count(voc.neutral_element)

    function_number = 0
    for char in voc.arity1symbols:
        function_number += state.reversepolish.count(char)

    powernumber = 0
    for char in state.reversepolish:
        if char == voc.power_number:
            powernumber += 1

    trignumber = 0
    for char in state.reversepolish:
        if char in voc.trignumbers:
            trignumber += 1

    explognumber = 0
    for char in state.reversepolish:
        if char in voc.explognumbers:
            explognumber += 1

    fnumber, deronenumber = 0, 0
    if config.use_derivative: #revoir
        for char in state.reversepolish:
            if voc.modescalar == 'noA':
                if char == 6:
                    fnumber = 1
                if char == 5:
                    deronenumber = 1
            else:
                if char == 5:
                    fnumber = 1
                if char == 4:
                    deronenumber = 1

    game = Game(voc, state)
    depth = game.getnumberoffunctions(state)

    return rms, state, alla, scalar_numbers, L, function_number, powernumber, trignumber, explognumber, fnumber, deronenumber, depth


def count_meta_features(voc, state):
    if state.reversepolish[-1] == voc.terminalsymbol:
        L = len(state.reversepolish) -1
    else:
        L = len(state.reversepolish)

    scalar_numbers = 0
    for char in voc.pure_numbers:
        scalar_numbers += state.reversepolish.count(char)

    scalar_numbers += state.reversepolish.count(voc.neutral_element)
    scalar_numbers += state.reversepolish.count(voc.true_zero_number)

    function_number = 0
    for char in voc.arity1symbols:
        function_number += state.reversepolish.count(char)

    powernumber = 0
    for char in state.reversepolish:
        if char == voc.power_number:
            powernumber += 1

    trignumber = 0
    for char in state.reversepolish:
        if char in voc.trignumbers:
            trignumber += 1

    explognumber = 0
    for char in state.reversepolish:
        if char in voc.explognumbers:
            explognumber += 1

    fnumber, deronenumber = 0, 0
    if config.use_derivative:
        for char in state.reversepolish:
            if voc.modescalar == 'noA':
                if char == 6:
                    fnumber=1
                if char == 5:
                    deronenumber = 1
            else:
                if char == 5:
                    fnumber=1
                if char == 4:
                    deronenumber = 1

    game = Game(voc, state)
    depth = game.getnumberoffunctions(state)


    return scalar_numbers, L, function_number, powernumber, trignumber, explognumber, fnumber, deronenumber, depth

# -------------------------------------------------------------------------- #
def exec(which_target, train_target, test_target, voc, iteration, gp, prefix):

    local_alleqs = {}
    monocore = True
    for i in range(iteration):
        #parallel cma
        print('')
        print('this is iteration', i)
        # this creates a pool of states or extends it before evaluation
        if config.auto_extent_size:
            pool = gp.extend_pool(i)
        else:
            pool = gp.extend_pool()
        print('pool creation/extension done')

        if voc.modescalar == 'A':
            pool_to_eval = []
            for state in pool:
                for u in range(len(config.training_target_list)):
                    pool_to_eval.append([test_target, voc, state, u])

            mp_pool = mp.Pool(config.cpus)
            asyncResult = mp_pool.map_async(evalme, pool_to_eval)
            results = asyncResult.get()
            # close it
            mp_pool.close()
            mp_pool.join()
            print('pool eval done')

            for result in results:
                # this is for the fact that an equation that has already been seen might return a better reward, because cmaes method is not perfect!
                if str(result[1].reversepolish) in local_alleqs:
                    if result[0] < local_alleqs[str(result[1].reversepolish)][0]:
                        local_alleqs.update({str(result[1].reversepolish): result})
                else:
                    local_alleqs.update({str(result[1].reversepolish): result})

        elif monocore == False:
            print('par pool')
            pool_to_eval = []
            print('verif', len(pool))
            for state in pool:
                for u in range(len(config.training_target_list)):
                    pool_to_eval.append([test_target, voc, state, u])

            mp_pool = mp.Pool(config.cpus)
            asyncResult = mp_pool.map_async(evalme, pool_to_eval)
            results = asyncResult.get()
            # close it
            mp_pool.close()
            mp_pool.join()
            print('pool eval done')

            for result in results:
                # this is for the fact that an equation that has already been seen might return a better reward, because cmaes method is not perfect!

                if str(result[1].reversepolish) in local_alleqs:
                    if result[0] < local_alleqs[str(result[1].reversepolish)][0]:
                        local_alleqs.update({str(result[1].reversepolish): result})
                else:
                    local_alleqs.update({str(result[1].reversepolish): result})
        else:
            results = []
            for state in pool:
                for u in range(len(config.training_target_list)):
                    scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, voc, test_target, 'test', u)
                    evaluate = Evaluatefit(state.formulas, voc, test_target, 'test', u)
                    evaluate.rename_formulas()
                    rms = evaluate.eval_reward_nrmse(alla)
                    scalar_numbers, L, function_number, powernumber, trignumber, explognumber,  fnumber, deronenumber, depth = count_meta_features(voc, state)
                    results.append([rms, state, alla, scalar_numbers, L, function_number, powernumber, trignumber, explognumber, fnumber, deronenumber, depth])
                if str(state.reversepolish) not in local_alleqs:
                    local_alleqs.update({str(state.reversepolish): results[-1]})

        results_by_bin = gp.bin_pool(results)

        # init
        if gp.QD_pool is None:
            gp.QD_pool = results_by_bin

        if voc.modescalar == 'A':
            type = '_a_'
        else:
            type = '_no_a_'


        newbin, replacements = gp.update_qd_pool(results_by_bin)


        save_qd_pool(gp.QD_pool, type)

        print('QD pool size', len(gp.QD_pool))
        print('alleqsseen', len(local_alleqs))


        # save results and print
        saveme = printresults(test_target, voc)
        tnumber = 999
        valrmse, bf = saveme.saveresults(newbin, replacements, i, gp.QD_pool, gp.maxa, tnumber, local_alleqs, prefix)

        if valrmse <config.termination_nmrse:
            del results
            print('early stopping')
            return 'es', gp.QD_pool, local_alleqs, i, valrmse, bf
        if len(local_alleqs) > 10000000:
            del results
            print(' stopping bcs too many eqs seen')
            return 'stop', gp.QD_pool, local_alleqs, i, valrmse, bf

    return None, gp.QD_pool, local_alleqs, i, valrmse, bf

# -----------------------------------------------#
def convert_eqs(qdpool, voc_a, voc_no_a, diff):
    # retrieve all the states, but replace integre scalars by generic scalar 'A' :
    allstates = []
    local_alleqs = {}
    for binid in qdpool:
        state = qdpool[binid][1]

        newstate = []
        for char in state.reversepolish:
            if char == voc_no_a.terminalsymbol:
                newstate.append(voc_a.terminalsymbol)
            elif char == voc_no_a.neutral_element:
                newstate.append(voc_a.pure_numbers[0])
            elif char == voc_no_a.true_zero_number:
                newstate.append(voc_a.pure_numbers[0])
            elif char in voc_no_a.pure_numbers:
                newstate.append(voc_a.pure_numbers[0])
            else:
                # shift everything : warning, works only if pure numbers are in the beginning of our dictionnaries!
                newstate.append(char - diff)

        allstates.append(newstate)

    # create the initial pool; now states have free scalars 'A'
    initpool = []

    # first simplfy the states, and remove infinities:
    for state in allstates:
        creastate = State(voc_a, state)
        if config.use_simplif:
            creastate = game_env.simplif_eq(voc_a, creastate)

        if voc_a.infinite_number not in creastate.reversepolish:
            if str(creastate.reversepolish) not in local_alleqs:
                local_alleqs.update({str(creastate.reversepolish): 1})
                initpool.append(creastate)

    del local_alleqs
    del allstates

    return initpool

# -----------------------------------------------#
def eval_previous_eqs(which_target, train_target, test_target, voc_a, initpool, gp, prefix):

    # init all eqs seen so far
    alleqs = {}

    pool_to_eval = []
    for state in initpool:
        for u in range(len(config.training_target_list)):
            pool_to_eval.append([test_target, voc_a, state,u])

    mp_pool = mp.Pool(config.cpus)
    print('how many states to eval : ', len(pool_to_eval))
    asyncResult = mp_pool.map_async(evalme, pool_to_eval)
    results = asyncResult.get()
    mp_pool.close()
    mp_pool.join()

    for result in results:
        alleqs.update({str(result[1].reversepolish): result})

    # bin the results
    results_by_bin = gp.bin_pool(results)
    gp.QD_pool = results_by_bin

    newbin, replacements = gp.update_qd_pool(results_by_bin)

    print('QD pool size', gp.QD_pool)
    print('alleqsseen', alleqs)

    # save results and print
    saveme = printresults(test_target, voc_a)
    tnumber = 999
    valrmse, bf = saveme.saveresults(newbin, replacements, -1, gp.QD_pool, gp.maxa, tnumber, alleqs, prefix)

    del mp_pool
    del asyncResult
    del results

    if valrmse < 0.00000000001:
        print('early stopping')
        return alleqs, gp.QD_pool, 'stop', valrmse, bf

    else:
        return alleqs, gp.QD_pool, None, valrmse, bf

# -----------------------------------------------#
def init_everything_else(which_target):
    # init targets
    if config.fromfile:
        train_targets = Target('train', which_target)
        test_targets = Target('test', which_target)
    else:
        train_targets = Target('train', None)
        test_targets = Target('test', None)

    # init dictionnaries
    voc_with_a = []
    voc_no_a = []
    for tar in train_targets:
        voc_with_a.append(Voc(tar, 'A'))
        voc_no_a.append(Voc(tar, 'noA'))

    #print('working with: ', voc_no_a.numbers_to_formula_dict)
    #print('and then with: ', voc_with_a.numbers_to_formula_dict)
    #print('with maximal size', voc_no_a.maximal_size)

    # useful
    sizea = []
    sizenoa = []
    for voc in voc_with_a:
        sizea.append(voc.numbers_to_formula_dict)
    for voc in voc_no_a:
        sizenoa.append(voc.numbers_to_formula_dict)
    diff = [sizenoa[i] - sizea[i] for i in range(len(sizea))]

    poolsize = config.qd_init_pool_size
    if config.MAX_DEPTH ==1:
        delete_ar1_ratio = 0.1
    elif config.MAX_DEPTH == 2:
        delete_ar1_ratio = 0.3
    else:
        delete_ar1_ratio = 0.8

    extend_ratio = config.extendpoolfactor
    p_mutate = 0.4
    p_cross = 0.8

    binl_no_a = voc_no_a.maximal_size # number of bins for length of an eq
    maxl_no_a = voc_no_a.maximal_size
    bina = maxl_no_a  # number of bins for number of free scalars
    maxa = bina
    binl_a = voc_with_a.maximal_size # number of bins for length of an eq
    maxl_a = voc_with_a.maximal_size
    binf = 160 # number of bins for number of fonctions
    maxf = 160
    new = 0
    binp = new  # number of bins for number of powers
    maxp = new
    bintrig = new # number of bins for number of trigonometric functions (sine and cos)
    maxtrig = new
    binexp = new # number of bins for number of exp-functions (exp or log)
    maxexp = new
    derzero, derone = 1 , 1 #absence ou presence de fo et ou de fo'
    addrandom = config.add_random

    return poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa, binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
           binp, maxp, bintrig, derzero, derone, maxtrig, binexp, maxexp, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff

# -----------------------------------------------#
def main(which_target):

    # init target, dictionnaries, and meta parameters
    poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa,  binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
    binp, maxp, bintrig,  derzero, derone,  maxtrig, binexp, maxexp, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff \
        = init_everything_else(which_target)
    prefix = str(int(10000000 * time.time()))
    stop = None
    if not config.skip_no_a_part:
        # init qd grid
        reinit_grid = False
        qdpool = init_grid(reinit_grid,  'QD_pool_no_a_.txt')

        # ------------------- step 1 -----------------------#
        gp = GP_QD(which_target, delete_ar1_ratio, p_mutate, p_cross, poolsize, voc_no_a,
                   extend_ratio, maxa, bina, maxl_no_a, binl_no_a, maxf, binf, maxp, binp, maxtrig, bintrig,  derzero, derone, maxexp, binexp,
                   addrandom, qdpool, None)

        if config.plot:
            #formm22 = '((((((d_x0_f0)*(49.746576))-(69.914573))/((0.209011)/(np.sin((x0)-(9.847268)))))+((np.sin((18.853701)))*((np.exp((10.663986)))+(-166.75883))))-' \
            #        '((-149.276147)-((f0)*((-30.305333)/(np.cos((-4.716783)))))))/(((np.sin((-7.777854)))-(63.70873))-((-65.709208)+(np.cos((x0)*(-14.665166)))))'
            #formm21  = '(((f0)*(np.exp((12.07992))))/(((np.cos(((54.871451)*(x0))-(11.075981)))+(-0.373353))-(((-78.102263)+((603.347966)+(np.exp((f0)*(-4.967299)))))*((x0)/((298.530341)-((f0)*(7.946753)))))))-((((np.cos((14.771823)/((x0)/(-11.829798))))/(2.340901))*(-38.751358))*((-633.53597)/(x0)))'
            #formm3 = '(x0)-((np.sin(((-265.747425)+((118.102741)/(x0)))/(x0)))*((((np.exp(x0))*((np.log((x0)+(x0)))-(x0)))*((x0)+(((np.exp((259.780332)))**((0.229831)+(np.sin((x0)*(-0.39444)))))*(np.sin((-1.569401)/((x0)/((x0)-(1.33829))))))))*(-0.878942)))'
            #formm1 = '-A[1]*f0-A[2]*d_x0_f0'
            #formm1 = '(x0)/(np.exp((26.550064)+((f0)**(-21.047723))))'
            #evall = Evaluatefit(formm1, voc_with_a, test_target, 'test')
            #print('re', evall.formulas)

            #evall.rename_formulas()
            #print(evall.formulas)
            #scalar_numbers, alla, rms = game_env.game_evaluate([1], evall.formulas, voc_no_a, test_target, 'test')
            #print('donne:', scalar_numbers, alla, rms)
            time.sleep(500)
        # run evolution :

        #try exact sol:

        iteration_no_a = config.iterationnoa
        stop, qdpool, alleqs_no_a, iter_no_a, valrmse, bf = exec(which_target, train_target, test_target, voc_no_a, iteration_no_a, gp, prefix)


        # ------------------- step 2 -----------------------#
        #if target has not already been found, stop is None; then launch evoltion with free scalars A:
        if stop is None:

            # convert noA eqs into A eqs:
            if config.saveqd:
                QD_pool = init_grid(False, 'QD_pool_a_.txt')
            else:
                initpool = convert_eqs(qdpool, voc_with_a, voc_no_a, diff)

                # reinit gp class with a:
                gp = GP_QD(which_target, delete_ar1_ratio, p_mutate, p_cross, poolsize,
                           voc_with_a, extend_ratio, maxa, bina, maxl_a, binl_a, maxf, binf, maxp, binp, maxtrig,
                           bintrig, derzero, derone, maxexp, binexp,
                           addrandom, initpool, None)
                alleqs_change_mode, QD_pool, stop, valrmse, bf = eval_previous_eqs(which_target, train_target,test_target,
                                                                                   voc_with_a,initpool, gp, prefix)

    if stop is None:
        if config.saveqd:
            QD_pool = init_grid(False, 'QD_pool_a_.txt')
        else:
            if config.skip_no_a_part:
                QD_pool = None

        gp = GP_QD(which_target, delete_ar1_ratio, p_mutate, p_cross, poolsize,
                   voc_with_a, extend_ratio, maxa, bina, maxl_a, binl_a, maxf, binf, maxp, binp, maxtrig,
                   bintrig, derzero, derone, maxexp, binexp,
                   addrandom, QD_pool, None)

        iteration_a = config.iterationa

        stop, qdpool, alleqs_a, iter_a, valrmse, bf = exec(which_target, train_target, test_target, voc_with_a, iteration_a, gp, prefix)


# -----------------------------------------------#
if __name__ == '__main__':

    # don't display any output
    noprint = False

    if noprint:
        import sys

        class writer(object):
            log = []

            def write(self, data):
                self.log.append(data)

        logger = writer()
        sys.stdout = logger
        sys.stderr = logger

    main(config.training_target_list)
