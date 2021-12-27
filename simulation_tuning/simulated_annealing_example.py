import random
import math

# https://ichi.pro/ko/python-eseo-simyulleisyeon-doen-eonilling-algolijeum-eul-guhyeonhaneun-bangbeob-188125677573792
def simulated_annealing(initial_state):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 9
    final_temp = .1
    alpha = 0.01

    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state

    while current_temp > final_temp:
        print("now on"+str(solution))
        neighbor = get_neighbors(solution)

        # Check if neighbor is best so far
        cost_diff = get_cost(current_state) - get_cost(neighbor)

        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
        # decrement the temperature
        current_temp -= alpha

    return solution, get_cost(solution)


def get_cost(state):
    """Calculates cost of the argument state for your solution."""
    print("cost of state"+str(state))
    print(state[0]**2+state[1]**2)
    return state[0]**2+state[1]**2


def get_neighbors(state):
    """Returns neighbors of the argument state for your solution."""
    p = random.uniform(0,1)
    if p<0.25:
        return [state[0]+0.01,state[1]]
    elif p<0.5:
        return [state[0] - 0.01, state[1]]
    elif p<0.75:
        return [state[0], state[1] + 0.01 ]
    else:
        return [state[0], state[1] - 0.01 ]


print("init")
print(get_cost([0.4,0.4]))
ANS = 10000
for i in range(100):
    ANS = min(simulated_annealing([0.4,0.4])[1],ANS)
print(ANS)