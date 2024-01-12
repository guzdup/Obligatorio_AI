import numpy as np
import gym
from acrobot_env_extended import AcrobotEnvExtended
import matplotlib.pyplot as plt
import pickle

#Ambiente:
env = AcrobotEnvExtended(render_mode='rgb_array')

#Angulos:                             Originalmente:
cost1_space = np.linspace(-1, 1, 10) # 5
sint1_space = np.linspace(-1, 1, 10) # 5
cost2_space = np.linspace(-1, 1, 20) # 10
sint2_space = np.linspace(-1, 1, 20) # 10
velt1_space = np.linspace(-12.57, 12.57, 40) # 30
velt2_space = np.linspace(-28.27, 28.27, 30) # 20

#Estado:
def get_state(obs):
    c1,s1,c2,s2,vt1,vt2 = obs
    c1_bin = np.digitize(c1, cost1_space)
    s1_bin = np.digitize(s1, sint1_space)
    c2_bin = np.digitize(c2, cost2_space)
    s2_bin = np.digitize(s2, sint2_space)
    vt1_bin = np.digitize(vt1, velt1_space)
    vt2_bin = np.digitize(vt2, velt2_space)
    return c1_bin, s1_bin,c2_bin,s2_bin,vt1_bin,vt2_bin

state = get_state(np.array([-0.4, 0.2, 0.3, 0.4, 0.5, 0.6]))

#Acciones:
actions = list(range(env.action_space.n))
actions

#Q:
Q = np.zeros((len(cost1_space)+1, len(sint1_space)+1, len(cost2_space)+1, len(sint2_space)+1, len(velt1_space)+1, len(velt2_space)+1, len(actions)))
Q

#Policy
def optimal_policy(state, Q):
    action = np.argmax(Q[state])
    return action

#Agente:
def epsilon_greedy_policy(state, Q, epsilon=0.1):
    explore = np.random.binomial(1, epsilon)
    if explore:
        action = env.action_space.sample()
        #print('explore')
    # exploit
    else:
        action = np.argmax(Q[state])
       # print('exploit')
        
    return action

#Reward: Jugar una iteracion
def play_one_iteration(): 
    obs,_ = env.reset()
    print(obs)
    done = False
    episode_reward = 0
    while not done:
        state = get_state(obs)
        action = epsilon_greedy_policy(state, Q, 0.5)
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward
        print('->', state, action, reward, obs, done)
        env.render()
    return episode_reward


def play_multiple_iterations(Q, num_iterations, epsilon, alpha, gamma):
    total_rewards = []

    for _ in range(num_iterations):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state = get_state(obs)
            action = epsilon_greedy_policy(state, Q, epsilon) 
            next_obs, reward, done, _, _ = env.step(action)

            # Actualizar el valor Q usando la ecuación de aprendizaje Q
            next_state = get_state(next_obs)
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]))
            obs = next_obs
            episode_reward += reward

        #total_rewards.append(episode_reward)

    #average_reward = np.mean(total_rewards)
    return episode_reward


# Uso de la función:
num_plays = 1 
#epsilon_value = 0.6  # Usar epsilon=0 para la fase de explotación
alpha_value = 0.7   # Tasa de aprendizaje
gamma_value = 0.9   # Factor de descuento


#Funcion que grafica el resultado explotado
def plot_exploit_rewards(Q, num_exploits):
    exploit_rewards = []  # Almacenar las recompensas promedio en cada iteración de "exploit"

    for i in range(num_exploits):
        # Realizar un "exploit" y calcular la recompensa promedio
        average_reward = testear_agente(Q, 20)
        exploit_rewards.append(average_reward)

    # Crear el gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_exploits + 1), exploit_rewards, marker='o')
    plt.title(f"Recompensa Promedio en Función de la Cantidad de Exploits: {num_exploits}")
    plt.xlabel("Cantidad de Exploits")
    plt.ylabel("Recompensa Promedio")
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()

def testear_agente(Q, num_iteraciones=20):
    total_rewards = []

    for _ in range(num_iteraciones):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state = get_state(obs)
            action = optimal_policy(state, Q)  # Utilizar la política óptima (Q max)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

    return episode_reward
    #promedio_recompensa = np.mean(total_rewards)
    #print(f"FINAL: Recompensa promedio en {num_iteraciones} partidas de prueba: {promedio_recompensa}")

# Guardar el agente en un archivo pickle
def save_agent(agent, filename):
    with open(filename, 'wb') as file:
        pickle.dump(agent, file)

# Comienzo de ejecucion
def exploration_phase(Q, epsilon, alpha, gamma, iterations):
    explorations_reward = []
    for _ in range(iterations):
        reward = play_multiple_iterations(Q, num_iterations=1, epsilon=epsilon, alpha=alpha, gamma=gamma)
        explorations_reward.append(reward)
    explorations_reward_mean = np.mean(explorations_reward)
    print(f"Exploration: Recompensa promedio en {iterations} partidas: {explorations_reward_mean}")
    return Q

def exploitation_phase(Q, num_exploits):
    testing_reward = []
    for _ in range(num_exploits):
        testear_agente
        testing_reward_mean = np.mean(testing_reward)
    print(f"Exploitation: Recompensa promedio en {num_exploits} partidas: {testing_reward_mean}")

# Parámetros iniciales
epsilon_value = 0.9
iterations_per_phase = 25
exploits_per_phase = 10

def exec():
    epsilon_value = 0.9
    num_exploits = 10 
    for i in range(80):
        explorations_reward = []
        testing_reward = []
        iterations = 25

        #-----Entrenamiento-----
        print(f"Iteración: {i}")
        print("Exploration")
        for episode in range(iterations):
            reward = play_multiple_iterations(Q, num_iterations=1, epsilon=epsilon_value, alpha=alpha_value, gamma=gamma_value)
            explorations_reward.append(reward)
        explorations_reward_mean = np.mean(explorations_reward)
        print(f"Exploration: Recompensa promedio en {iterations} partidas: {explorations_reward_mean}, con epsilon: {epsilon_value}")

        #-----Exploracion-----
        print("Exploitation")
         
        for _ in range(num_exploits):
            reward = testear_agente(Q, num_iteraciones=20)
            testing_reward.append(reward)
        testing_reward_mean = np.mean(testing_reward)
        print(f"Exploitation: Recompensa promedio en {num_exploits} partidas: {testing_reward_mean}")

        # Actualizar epsilon
        if i % 5 == 0 and i != 0:
            epsilon_value -= 0.02
        if i % 20 == 0:
            num_exploits += 10
        
        agent1 = [Q]
        #print(Q)
        save_agent(agent1, "agente1.pkl")

exec()

pickel = "agente1.pkl"

# Cargar el objeto desde el archivo .pkl
#with open(pickel, 'rb') as file:
 #   objeto_cargado = pickle.load(file)

# Imprimir el contenido del objeto
#print(objeto_cargado)

#agent1 = [Q]
#print(Q)
#save_agent(agent1, "agente1.pkl")

# Cargar el agente desde un archivo pickle
def load_agent(filename):
    with open(filename, 'rb') as file:
        agent = pickle.load(file)
    return agent
