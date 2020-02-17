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
def save_qd_pool(pool):
    timeur = int(time.time()*1000000)
    if config.uselocal:
        file_path = 'QD_pool' +  '.txt'
    else:
        file_path = '/home/user/results/QD_pool' + str(timeur) + '.txt'

    with open(file_path, 'wb') as file:
        pickle.dump(pool, file)
        file.close()

# -------------------------------------------------------------------------- #
def evalme(onestate):
    train_targets, voc, state, u, look_for = onestate[0], onestate[1], onestate[2], onestate[3], onestate[4]

    results = []
    scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, voc, train_targets, 'train',u, look_for)
    results.append([rms, scalar_numbers, alla])

    if config.tworunsineval:
        scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, voc, train_targets, 'train',u, look_for)
        results.append([rms, scalar_numbers, alla])
        if results[0][0] <= results[1][0]:
            rms, scalar_numbers, alla = results[0]
        else:
            rms, scalar_numbers, alla = results[1]

    if state.reversepolish[-1] == voc.terminalsymbol:
        L = len(state.reversepolish) -1
    else:
        L = len(state.reversepolish)

    function_number = 0
    for char in voc.arity1symbols:
        function_number += state.reversepolish.count(char)

    powernumber = 0
    for char in state.reversepolish:
        if char == voc.power_number:
            powernumber += 1

    fnumber, deronenumber = 0, 0
    # if config.use_derivative: #todo revoir
    #     for char in state.reversepolish:
    #         if voc.modescalar == 'noA':
    #             if char == 6:
    #                 fnumber = 1
    #             if char == 5:
    #                 deronenumber = 1
    #         else:
    #             if char == 5:
    #                 fnumber = 1
    #             if char == 4:
    #                 deronenumber = 1

    game = Game(voc, state)
    depth = game.getnumberoffunctions(state)

    return rms, state, alla, scalar_numbers, L, function_number, powernumber, fnumber, deronenumber, depth

# -------------------------------------------------------------------------- #
def exec(train_targets, test_targets, u, voc, iteration, gp, prefix, look_for, calculus_mode):

    local_alleqs = {}
    for i in range(iteration):
        #parallel cma
        print('')
        print('this is iteration', i)
        # this creates or extends a pool of states before evaluation
        pool = gp.extend_pool()
        print('pool creation/extension done')

        pool_to_eval = []
        for state in pool:
            pool_to_eval.append([train_targets, voc, state, u, look_for])

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

        results_by_bin = gp.bin_pool(results)

        # init
        if gp.QD_pool is None:
            gp.QD_pool = results_by_bin


        newbin, replacements = gp.update_qd_pool(results_by_bin)
        save_qd_pool(gp.QD_pool)

        print('QD pool size', len(gp.QD_pool))
        print('alleqsseen', len(local_alleqs))


        # save results and print
        saveme = printresults(train_targets, voc)
        tnumber = 999
        valrmse, bf = saveme.saveresults(newbin, replacements, i, gp.QD_pool, gp.maxa, tnumber, local_alleqs, prefix, u, look_for)

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
def main(params, train_targets, test_targets, u, look_for, calculus_mode, maximal_size):

    # init target, dictionnaries, and meta parameters
    poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa,  binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
    binp, maxp,  derzero, derone, addrandom, voc = params

    prefix = str(int(10000000 * time.time()))
    gp = GP_QD(delete_ar1_ratio, p_mutate, p_cross, poolsize, voc,
               extend_ratio, maxa, bina, maxl_no_a, binl_no_a, maxf, binf, maxp, binp, derzero, derone,
               addrandom, calculus_mode, maximal_size, None, None)

    iteration_a = config.iterationa

    stop, qdpool, alleqs_a, iter_a, valrmse, bf = exec(train_targets, test_targets, u, voc, iteration_a, gp, prefix, look_for, calculus_mode)

