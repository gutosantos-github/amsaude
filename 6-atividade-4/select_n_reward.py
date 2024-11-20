import random
from classDiabEnv import *
from states_n_actions import *
import numpy as np

# Função para selecionar ação usando a política epsilon-greedy
def select_action(state, q_table, epsilon):
    #probabilidade entre 0 e 1
    if random.uniform(0, 1) < epsilon:
        return random.choice([INSULIN, EAT, EXERCISE, DRINK_WATER, REST])  # Explorar (ação aleatória)
    else:
        return np.argmax(q_table[state])  # Exploração do melhor valor conhecido

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title('Recompensas por Episódio')
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa')
    plt.show()