import numpy as np
import random
import matplotlib.pyplot as plt
from select_n_reward import *
from states_n_actions import *


class DiabetesEnv:
    def __init__(self):
        self.state = random.choice([GLUCOSE_LOW, GLUCOSE_NORMAL, GLUCOSE_HIGH])

    def reset(self):
        # Reinicia o ambiente em um estado aleatório
        self.state = random.choice([GLUCOSE_LOW, GLUCOSE_NORMAL, GLUCOSE_HIGH])
        return self.state

    def step(self, action):
        # Transição entre estados baseada na ação tomada
        if action == INSULIN:
            next_state = max(GLUCOSE_LOW, self.state - 1)
        elif action == EAT:
            next_state = min(GLUCOSE_HIGH, self.state + 1)
        elif action == EXERCISE:
            next_state = max(GLUCOSE_LOW, self.state - 1)
        elif action == DRINK_WATER:
            # Beber água estabiliza, não altera o estado
            next_state = self.state
        elif action == REST:
            # Descansar pode aumentar glicose se já estiver em um nível baixo ou normal
            next_state = min(GLUCOSE_HIGH, self.state + 1)

        # Recompensas: Manter nível de glicose normal é positivo
        reward = 10 if next_state == GLUCOSE_NORMAL else -1
        done = False

        # Atualiza o estado do ambiente
        self.state = next_state
        return next_state, reward, done, {}