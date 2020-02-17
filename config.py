# _________ define voc ____________
all_fonctions = ['np.cos(', 'np.tan(', 'np.exp(', 'np.log(', 'np.sqrt(', 'np.sinh(', 'np.cosh(', 'np.tanh(', 'np.arcsin(', 'np.arctan(']
small_set = ['np.sin(', 'np.sqrt(', 'np.exp(', 'np.log(']#,'np.arcsin(', 'np.arctan(']

fonctions = small_set #make your choice here
#list_scalars = ['minuso_ne', 'minust_wo', 'minust_hree', 'minusf_our', 'minusf_ive', 'one', 'two', 'three', 'four', 'five']
list_scalars = ['one', 'two']

operators = ['+', '-', '*', '/']

#____________termination stuff_______________
iterationa = 100
iterationnoa = 100#2015
termination_nmrse = 1e-6

#_________________Taget related_______________#


# how many nested functions I authorize
MAX_DEPTH = 1
# power is taken only to a real number : avoid stuff like exp(x)^(x exp(x)) !!
only_scalar_in_power = True

fromfile = False
multiple_training_targets = True
if fromfile:

    training_target_list = ['GWfiles\wh_2.txt', 'GWfiles\wl_2.txt']
    #training_target_list = ['OHAs/OH_0.5_-0.4_0.5_1.4.txt', 'OHAs/OH_0.7_-1.3_1.5_0.7.txt']
    maxsize = 15
else:
    training_target_list = 0


# _______________ QD algo related ______________
auto_extent_size = False
add_random = True
skip_no_a_part = False
saveqd = False
use_simplif = False


import numpy as np
def get_size(iteration):
    internal_nodes = [15]
    programmation = [100]
    cumuprog = np.cumsum(programmation)
    index = np.where(iteration <= cumuprog)[0]
    index = index[0]
    size = 2*internal_nodes[index]+1
    return size

qd_init_pool_size = 500
extendpoolfactor = 1
#which_target = 'fig1-waveform-H_phase2_1.txt'

plot = False
tworunsineval = False
popsize = '10'
timelimit = '7*N'

# reward decreases if there are too many pure scalar parameters, given by:
parsimony = 1500
parsimony_cost = 0.02

# max number of scalars allowed in a rd eq:
max_A_number = 1200

# simplif on the fly

# -------------------- reward related -------------------------- #
#after some tests, its better to use both the distance cost AND the derivative cost
usederivativecost = 0  #or 0

#misc
uselocal = True
cpus = 8

trytrick = False
tryoscamorti = False

