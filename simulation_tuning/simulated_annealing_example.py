import random
import math

# https://ichi.pro/ko/python-eseo-simyulleisyeon-doen-eonilling-algolijeum-eul-guhyeonhaneun-bangbeob-188125677573792

def out(state):
    if abs(state[0])>1 or abs(state[1])>1:
        return True
    else:
        return False
def simulated_annealing(initial_state):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 900
    final_temp = .1
    alpha = 0.01

    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state

    min_state = initial_state

    while current_temp > final_temp:
        #print("now on"+str(solution))
        neighbor = get_neighbors(solution)

        # Check if neighbor is best so far
        cost_diff = get_cost(current_state) - get_cost(neighbor)

        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
            if get_cost(solution)<get_cost(min_state):
                min_state = solution
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
        # decrement the temperature
        current_temp -= alpha

    return min_state, get_cost(min_state)


def get_cost(state):
    """Calculates cost of the argument state for your solution."""
    #print("cost of state"+str(state))
    #print(state[0]**2+state[1]**2)
    return state[0]**3+state[1]**3


def get_neighbors(state):
    """Returns neighbors of the argument state for your solution."""
    '''
    p = random.uniform(0,1)
    if p<0.25:
        return [state[0]+0.01,state[1]]
    elif p<0.5:
        return [state[0] - 0.01, state[1]]
    elif p<0.75:
        return [state[0], state[1] + 0.01 ]
    else:
        return [state[0], state[1] - 0.01 ]
    '''
    ret = random.choice([[state[0]+0.01,state[1]],[state[0] - 0.01, state[1]],[state[0], state[1] + 0.01 ],[state[0], state[1] - 0.01 ]])
    while out(ret):
        ret = random.choice([[state[0] + 0.01, state[1]], [state[0] - 0.01, state[1]], [state[0], state[1] + 0.01],
                             [state[0], state[1] - 0.01]])
    return ret




print("init")
x0 = [1,1]
print(get_cost(x0))
ANS = 10000
for i in range(100):
    print("in"+str(i))
    t = simulated_annealing(x0)[1]
    print(t)
    ANS = min(t,ANS)
print(ANS)