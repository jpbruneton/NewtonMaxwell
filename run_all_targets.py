import run_one_target
import config
from Targets import Target, Voc

# -----------------------------------------------#
def init_everything(train_target, test_target):
    # init dictionnaries
    voc_with_a = Voc(train_target, 'A')
    voc_no_a = Voc(train_target, 'noA')

    print('for target name', train_target[0])
    print('we work with no A voc: ', voc_no_a.numbers_to_formula_dict)
    print('and then with A voc: ', voc_with_a.numbers_to_formula_dict)
    print('and with maximal size', voc_no_a.maximal_size)

    # useful
    diff = len(voc_no_a.numbers_to_formula_dict) - len(voc_with_a.numbers_to_formula_dict)
    # metaparameters
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

    params = [poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa, binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
           binp, maxp, bintrig, derzero, derone, maxtrig, binexp, maxexp, addrandom, voc_with_a, voc_no_a, diff]
    return params

# -----------------------------------------------#
def kill_print():
    import sys
    class writer(object):
        log = []

        def write(self, data):
            self.log.append(data)

    logger = writer()
    sys.stdout = logger
    sys.stderr = logger

def load_targets(filenames_train, filenames_test):
    train = Target(filenames_train).targets
    test = Target(filenames_test).targets

    # flatten all targets
    train_targets = [train[0][0]]
    test_targets = [test[0][0]]

    funcs = []
    fder = []
    sder = []
    for u in range(len(train)):
        funcs.extend(train[u][1])
        fder.extend(train[u][2])
        sder.extend(train[u][3])
    train_targets.extend([funcs, fder, sder])

    funcs = []
    fder = []
    sder = []
    for u in range(len(test)):
        funcs.extend(test[u][1])
        fder.extend(test[u][2])
        sder.extend(test[u][3])
    test_targets.extend([funcs, fder, sder])
    return train_targets, test_targets

# -----------------------------------------------#
if __name__ == '__main__':
    # don't display any output
    noprint = False
    if noprint:
        kill_print()

    filenames_train = ['data_loader/x1_train(t).csv','data_loader/x2_train(t).csv']
    filenames_test = ['data_loader/x1_test(t).csv','data_loader/x2_test(t).csv']


    train_targets, test_targets = load_targets(filenames_train, filenames_test)
    # train et test de la forme [t #l'array de la variable, [les n targets], [les n der premieres], [les n der secondes]] avec un t commun


    # on résout une par une
    for u in range(len(train_targets)):
        train_target = train_targets[u]
        test_target = test_targets[u]
        params = init_everything(train_target, test_target)

        if train_target[1] == 1:#une seule variable t, genre x(t)
            run_one_target.main(params, train_target, test_target, 'der_sec')

        else: # j'ai un champ, je boucle pour chercher son der x dery, derz, dert
            run_one_target.main(params, train_target, test_target, 'dx')
            run_one_target.main(params, train_target, test_target, 'dy')
            run_one_target.main(params, train_target, test_target, 'dz')
            run_one_target.main(params, train_target, test_target, 'dt')

