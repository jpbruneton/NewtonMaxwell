import numpy as np
import copy
import random
import config
from operator import itemgetter
from generate_offsprings import generate_offsprings
from Evaluate_fit import Evaluatefit
import time
import game_env
import multiprocessing as mp
from multiprocessing import Pool


# ============================  QD version ====================================#

# ---------------------------------------------------------------------------- #
class GP_QD():

    def __init__(self, delete_ar1_ratio, p_mutate, p_cross, poolsize, voc,
                  extend_ratio, maxa, bina, maxl, binl, maxf, binf, maxp, binp, derzero, derone, addrandom,
                  calculculus_mode, maximal_size, qdpool, pool = None):

        self.calculus_mode = calculculus_mode
        self.usesimplif = config.use_simplif
        self.p_mutate = p_mutate
        self.delete_ar1_ratio = delete_ar1_ratio
        self.p_cross= p_cross
        self.poolsize = poolsize
        self.pool = pool
        self.QD_pool = qdpool
        self.pool_to_eval = []
        self.maximal_size = maximal_size
        self.extend = extend_ratio
        self.addrandom = addrandom
        self.voc = voc
        self.maxa = maxa
        self.bina = bina
        self.maxl = maxl
        self.binl = binl
        self.maxf = maxf
        self.binf = binf
        self.maxp = maxp
        self.binp = binp
        self.derzero = derzero
        self.derone = derone
        self.maxder = 1
        self.smallstates =[]

    # ----------------------
    def par_crea(self, task):
        np.random.seed(task)
        newgame = game_env.randomeqs(self.voc)
        return newgame.state

    # ---------------------------------------------------------------------------- #
    #creates or extend self.pool
    def extend_pool(self, iteration = None):

        if self.pool == None and self.QD_pool is None:
            self.pool = []
            tasks = range(0, self.poolsize)
            # print(tasks)
            mp_pool = mp.Pool(config.cpus)
            asyncResult = mp_pool.map_async(self.par_crea, tasks)
            results = asyncResult.get()
            mp_pool.close()
            mp_pool.join()
            for state in results:
                if self.voc.infinite_number not in state.reversepolish:
                    self.pool.append(state)
            del mp_pool
            return self.pool


        else:

            gp_motor = generate_offsprings(self.delete_ar1_ratio, self.p_mutate, self.p_cross, self.maximal_size, self.voc, self.calculus_mode)
            all_states = []
            small_states=[]
            for bin_id in self.QD_pool:
                all_states.append(self.QD_pool[str(bin_id)][1])
                if eval(bin_id)[1] < self.maximal_size - 10:
                    small_states.append(self.QD_pool[str(bin_id)][1])

            self.smallstates = small_states
            print('euh1', len(all_states))

            # +add new rd eqs for diversity. We add half the qd_pool size of random eqs
            if self.addrandom or self.maximal_size < 10:
                toadd = int(len(self.QD_pool)/2)
                c=0
                ntries = 0

                st = time.time()

                while c < toadd and ntries < 2000:
                    newgame = game_env.randomeqs(self.voc)
                    if self.voc.infinite_number not in newgame.state.reversepolish:
                        all_states.append(newgame.state)
                        c += 1
                    ntries += 1
                print('toi complet', time.time() -st)

            print('euh2', len(all_states))
            #print('trying', ntries, 'vs', toadd)
            ts = time.time()
            print('sizetocross', len(self.QD_pool))
            #then mutate and crossover
            newpool = []
            count=0

            treshold = max(int(self.extend*len(self.QD_pool)),config.qd_init_pool_size)
            treshold = int(self.extend * len(self.QD_pool))

            while len(newpool) < treshold and count < 400000:
                index = np.random.randint(0, len(all_states))
                state = all_states[index]
                u = random.random()

                if u <= self.p_mutate:
                    count += 1
                    if self.calculus_mode == 'scalar':
                        s, mutatedstate = gp_motor.mutate(state)
                    else:
                        s, mutatedstate = gp_motor.vectorial_mutation(state)

                    #if str(mutatedstate.reversepolish) not in alleqs:
                    newpool.append(mutatedstate)

                elif u <= self.p_cross:
                    count += 2

                    index = np.random.randint(0, len(all_states))
                    otherstate = all_states[index]  # this might crossover with itself : why not!
                    if self.calculus_mode == 'scalar':
                        success, state1, state2 = gp_motor.crossover(state, otherstate)
                    else:
                        success, state1, state2 = gp_motor.vectorial_crossover(state, otherstate)

                    if success:
                        #if str(state1.reversepolish) not in alleqs:
                        newpool.append(state1)
                        #if str(state2.reversepolish) not in alleqs:
                        newpool.append(state2)

                else:  # mutate AND cross
                    count += 2

                    index = np.random.randint(0, len(all_states))
                    to_mutate = copy.deepcopy(all_states[index])
                    if self.calculus_mode == 'scalar':
                        s, prestate1 = gp_motor.mutate(state)
                        s, prestate2 = gp_motor.mutate(to_mutate)
                        suc, state1, state2 = gp_motor.crossover(prestate1, prestate2)
                    else:
                        s, prestate1 = gp_motor.vectorial_mutation(state)
                        s, prestate2 = gp_motor.vectorial_mutation(to_mutate)
                        suc, state1, state2 = gp_motor.vectorial_crossover(prestate1, prestate2)
                    if suc:
                        #if str(state1.reversepolish) not in alleqs:
                        newpool.append(state1)
                        #if str(state2.reversepolish) not in alleqs:
                        newpool.append(state2)

            print('avgtime', (time.time()-ts))
            #update self.pool
            self.pool = newpool
            print('yo', len(newpool))


            return self.pool

    # ---------------------------------------------------------------------------- #
    # bin the results
    def bin_pool(self, results):

        results_by_bin = {}
        ct=0
        cif=0
        cnotif=0
        # rescale and print which bin
        for oneresult in results:
            rms, state, allA, Anumber, L, function_number, powernumber, fnumber, deronenumber, depth = oneresult
            #print(reward, state, allA, Anumber, L, function_number, powernumber, trignumber, explognumber)
            #if state.reversepolish[-1] == 1:
            #    L = L - 1

            #needs to be computed if mode 'noA'
            #if self.voc.modescalar == 'noA':
            #    Anumber = 0
            #    for char in state.reversepolish:
            #        if char in self.voc.pure_numbers:
            #            Anumber+=1

            if Anumber >= self.maxa:
                bin_a = self.maxa
            else:
                bins_for_a = np.linspace(0, self.maxa, num=self.maxa+1)
                for i in range(len(bins_for_a) -1):
                    if Anumber >= bins_for_a[i] and Anumber < bins_for_a[i + 1]:
                        bin_a = i

            if L >= self.maxl:
                bin_l = self.maxl
            else:
                bins_for_l = np.linspace(0, self.maxl, num=self.maxl+1)
                for i in range(len(bins_for_l) - 1):
                    if L >= bins_for_l[i] and L < bins_for_l[i + 1]:
                        bin_l = i

            if function_number >= self.maxf:
                bin_f = self.maxf
            else:
                bins_for_f = np.linspace(0, self.maxf, num = self.maxf+1)
                for i in range(len(bins_for_f) - 1):
                    if function_number >= bins_for_f[i] and function_number < bins_for_f[i + 1]:
                        bin_f = i

            if fnumber >= self.maxder:
                bin_fzero = self.maxder
            else:
                bin_fzero = np.linspace(0, self.maxder, num = self.maxder+1)
                for i in range(len(bin_fzero) - 1):
                    if fnumber >= bin_fzero[i] and fnumber < bin_fzero[i + 1]:
                        bin_fzero = i

            if deronenumber >= self.maxder:
                bin_fone = self.maxder
            else:
                bin_fone = np.linspace(0, self.maxder, num = self.maxder+1)
                for i in range(len(bin_fone) - 1):
                    if deronenumber >= bin_fone[i] and deronenumber < bin_fone[i + 1]:
                        bin_fone = i


            bin_for_d = np.linspace(0, config.MAX_DEPTH, num=config.MAX_DEPTH + 2)
            for i in range(len(bin_for_d) - 1):
                if depth >= bin_for_d[i] and deronenumber < bin_for_d[i + 1]:
                    bin_d = i

            #exclude trivial cases in my grid
            bin_d =0

            if bin_d ==0:
                bin_d = 1
            bin_d =0

            if powernumber >= self.maxp:
                bin_p = self.maxp
            else:
                bins_for_p = np.linspace(0, self.maxp, num=self.maxp + 1)
                for i in range(len(bins_for_p) - 1):
                    if powernumber >= bins_for_p[i] and powernumber < bins_for_p[i + 1]:
                        bin_p = i


            ct+=1
            if str([bin_a, bin_l, bin_p, bin_f, bin_fzero, bin_fone, bin_d]) not in results_by_bin:
                cif+=1
                if rms <2:
                    results_by_bin.update({str([bin_a, bin_l, bin_p, bin_f, bin_fzero, bin_fone, bin_d]): [rms, state, allA]})
                    #print('new elem', state.formulas)
                    #print([bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f, bin_fzero, bin_fone, bin_d])
                    #print('depth is', depth)
                #print(results_by_bin)
            else:
                cnotif+=1
                #print('happening', [bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f])
                #time.sleep(.2)
                #print(state.formulas)
                prev_rms = results_by_bin[str([bin_a, bin_l, bin_p, bin_f, bin_fzero, bin_fone, bin_d])][0]
                #print('prev', results_by_bin[str([bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f])])
                #print('prev', results_by_bin[str([bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f])][1].formulas)
                if rms < prev_rms:
                    results_by_bin.update({str([bin_a, bin_l, bin_p, bin_f, bin_fzero, bin_fone, bin_d]): [rms, state, allA]})
        #print(results_by_bin)
        #print('me', len(results_by_bin))
        print(ct, cif, cnotif)
        return results_by_bin

    # ---------------------------------------------------------------------------- #
    #updtae qd_pool according to new results
    def update_qd_pool(self, newresults_by_bin):
        newbin = 0
        replacement = 0

        for binid in newresults_by_bin:
            if binid not in self.QD_pool:
                self.QD_pool.update({binid: newresults_by_bin[binid]})
                newbin += 1

            else:
                prev_rms = self.QD_pool[binid][0]
                rms = newresults_by_bin[binid][0]
                if rms < prev_rms:
                    self.QD_pool.update({binid: newresults_by_bin[binid]})
                    replacement += 1
        print('news', newbin, replacement)
        return newbin, replacement

# ========================   end class gp_qd ====================== #

class printresults():

    def __init__(self, target, voc, calculusmode):
        self.target = target
        self.voc = voc
        self.calculusmode = calculusmode

    # ---------------------------------------------- |
    # to print understandable results
    def finalrename(self, bestform, A):

        formula = bestform

        As = [int(1000000*x)/1000000 for x in A]

        if As != []:

            rename = ''
            A_count = 0
            for char in formula:
                if char == 'A':
                    rename += 'A[' + str(A_count) + ']'
                    A_count += 1
                else:
                    rename += char

            rename = rename.replace('np.', '')
            rename = rename.replace('x0', 't')
            rename = rename.replace('x1', 'y')
            rename = rename.replace('x2', 'z')

            #handle le plus one
            if A_count < len(As):
                print('this is obsolete')
                rename += '+ A[' + str(A_count) + ']'

                for i in range(A_count + 1):
                    to_replace = 'A[' + str(i) + ']'
                    replace_by = '(' + str(As[i]) + ')'
                    rename = rename.replace(to_replace, replace_by)
            else:

                for i in range(A_count):
                    to_replace = 'A[' + str(i) + ']'
                    replace_by = '(' + str(As[i]) + ')'
                    rename = rename.replace(to_replace, replace_by)
        else:
            formula = formula.replace('np.', '')
            formula = formula.replace('x0', 'x')
            formula = formula.replace('x1', 'y')
            formula = formula.replace('x2', 'z')
            rename = formula

        return rename


    def saveresults(self, newbin, replacements, i, QD_pool, maxa, target_number, alleqs, prefix, u, look_for):

        # rank by number of free parameters
        bests = []
        best_simplified_formulas = []


        for a in range(maxa):
            eqs = []
            for bin_id in QD_pool:
                anumber = int(bin_id.split(',')[0].replace('[', ''))
                if anumber == a:
                    eqs.append(QD_pool[str(bin_id)])

            if len(eqs) > 0:
                sort = sorted(eqs, key=itemgetter(0), reverse=False)
                thebest = sort[0]
                thebestformula = thebest[1].formulas
                thebest_as = thebest[2]
                #simple = game_env.simplif_eq(self.voc, thebest[1])
                #best_simplified_formulas.append(simple.formulas)
                bests.append([thebest[0], self.finalrename(thebestformula, thebest_as)])

        # best of all
        all_states = []
        for bin_id in QD_pool:
            all_states.append(QD_pool[str(bin_id)])

        rank = sorted(all_states, key=itemgetter(0), reverse=False)
        best_state = rank[0][1]
        with_a_best = rank[0][2]
        best_formula = best_state.formulas
        bestreward = rank[0][0]
        print('wtf', bestreward)

        if np.isnan(bestreward) or np.isinf(bestreward):
            bestreward=100000000
        evaluate = Evaluatefit(best_formula, self.voc, self.target, 'train', u, look_for)
        evaluate.rename_formulas()

        if self.calculusmode == 'scalar':
            validation_reward = evaluate.eval_reward_nrmse(with_a_best)
        else:
            validation_reward = evaluate.eval_reward_nrmse_vectorial(with_a_best)

        if validation_reward > 100000000:
            validation_reward = 100000000

        if np.isnan(validation_reward) or np.isinf(validation_reward):
            validation_reward = 100000000

        useful_form = self.finalrename(best_formula, with_a_best)

        if bestreward < config.termination_nmrse:
            print(best_formula, with_a_best)
            print(evaluate.formulas)
            print(validation_reward)

        if bestreward < config.termination_nmrse:
            validation_reward = 0.
        #other statistics
        avgreward = 0
        for x in rank:
            reward = x[0]
            if np.isnan(reward) or np.isinf(reward):
                reward=100000000
            avgreward += reward

        avgreward = avgreward / len(rank)
        #avg validation reward:
        avg_validation_reward = 0
        for x in rank:
            state = x[1]
            with_a = x[2]

            formula = state.formulas

            evaluate = Evaluatefit(formula, self.voc, self.target, 'train',u, look_for)
            evaluate.rename_formulas()
            if self.calculusmode == 'scalar':
                avg_validation_reward += evaluate.eval_reward_nrmse(with_a)
            else:
                avg_validation_reward += evaluate.eval_reward_nrmse_vectorial(with_a)

        avg_validation_reward /= len(rank)

        timespent = time.time() - eval(prefix)/10000000
        if config.uselocal:
            filepath = './results/' + prefix + 'results_target_' + str(target_number) + '.txt'
        else:
            filepath = '/home/user/results/'+ prefix+ 'results_target_' + str(target_number) + '.txt'
        with open(filepath, 'a') as myfile:

            myfile.write('iteration ' + str(i) + ': we have seen ' + str(len(alleqs)) + ' different eqs')
            myfile.write("\n")
            myfile.write(
                'QD pool size: ' + str(len(QD_pool)) + ', newbins: ' + str(newbin) + ' replacements: ' + str(
                    replacements))
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('new avg training reward: ' + str(int(1000 * avgreward) / 1000))
#            myfile.write(' new avg validation reward: ' + str(int(1000 * avg_validation_reward) / 1000))
            myfile.write("\n")

            myfile.write('best reward: ' + str(bestreward) + ' with validation reward: ' + str(validation_reward))
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('best eq: ' + str(useful_form) + ' ' + str(best_formula) + ' ' + str(with_a_best))
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('and bests eqs by free parameter number are:')
            myfile.write("\n")
            myfile.write(str(bests))
            myfile.write("\n")
            myfile.write("\n")
            myfile.write('time spent (in secs):' + str(timespent))
            myfile.write("\n")
            myfile.write("\n")
            myfile.write("---------------=============================----------------")
            myfile.write("\n")

            myfile.close()

        return validation_reward, best_simplified_formulas
