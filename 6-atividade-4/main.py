from classDiabEnv import *
from qlearning import *
from select_n_reward import *
from evaluate_agent import *

# VERSÃO MODIFICADO FINAL COM FUNÇÃO "EVALUATE_AGENT" E AÇÕES "DRINK WATER" E "REST"
# 19NOV2024 ÀS 08:11H

# Executar o treinamento
if __name__ == '__main__':
    env = DiabetesEnv()
    q_table, rewards = q_learning(env)

    # Exibir Q-table final
    print("Q-table Final:")
    print(q_table)

    # Plotar recompensas
    plot_rewards(rewards)

    # Avaliar o agente
    max_steps_per_episode = 100
    num_episodes = 1000
    eval_seed = np.random.randint(0, 1000, size=num_episodes)  # cria uma lista de seeds para cada episódio

    mean_reward, std_reward = evaluate_agent(env, max_steps_per_episode, num_episodes, q_table, eval_seed)
    print("\nAvaliação do Agente: ")
    print(f"Mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}")