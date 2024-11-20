import numpy as np
from select_n_reward import select_action
from classDiabEnv import n_states, n_actions

# Algoritmo Q-learning para otimização de tratamento
def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.99,
               max_steps_per_episode=100):
    # Inicializar a Q-table com zeros
    q_table = np.zeros((n_states, n_actions))
    rewards = []

    for episode in range(num_episodes):
        # Estado inicial
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Seleciona uma ação com base na política epsilon-greedy
            action = select_action(state, q_table, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Calcula o TD Target para atualização da Q-table
            td_target = reward + gamma * np.max(q_table[next_state])
            q_table[state, action] = q_table[state, action] + alpha * (td_target - q_table[state, action])

            # Atualiza o estado e a recompensa acumulada
            state = next_state
            total_reward += reward

            if done:
                break

        # Exibir progresso a cada 100 episódios
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1:5d}  Eps: {epsilon:.4f}  Total Reward: {total_reward:.4f}")

        rewards.append(total_reward)
        epsilon = max(0.01, epsilon * epsilon_decay)

    return q_table, rewards