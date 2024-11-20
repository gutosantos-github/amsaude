import numpy as np


def evaluate_agent(env, max_steps_per_episode, num_episodes, q_table, seed):
    """
    :param env: ambiente de interação do agente
    :param max_steps_per_episode: número máximo de passos definido pelo professor
    :param num_episodes: número de episódios de avaliação do agente
    :param q_table: mapeamento dos estados para as ações e as recompensas
    :param seed: reproducibilidade
    :return: retorno
    """
    episode_rewards = []
    for episode in range(num_episodes):
        if seed is not None and len(seed) > 0:
            np.random.seed(seed[episode])  # Define a seed para cada episódio
        state = env.reset()  # Reinicializa o ambiente
        done = False
        total_rewards_ep = 0

        for step in range(max_steps_per_episode):
            # Ação com maior valor de recompensa na Q-table
            action = np.argmax(q_table[state][:])
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward