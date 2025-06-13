import torch
import random
from src import DDoSEnvironment, DQNAgent


MODEL_PATH = './results/'

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"현재 사용 중인 장치: {device}")

state_size = 8
action_size = 10
steps = 500

saved_state_dict = torch.load(MODEL_PATH)
agent = DQNAgent(state_size, action_size,device=device)

agent.policy_net.load_state_dict(saved_state_dict)
agent.policy_net.eval()
is_env_change = False

r1 = 5; r2 = 2; r3 = 3; r4 = 3

if is_env_change == True:
    r1 = random.randint(2,6)
    r2 = random.randint(2,6)
    r3 = random.randint(2,6)
    r4 = random.randint(2,6)
    print(f'구조 : [{r1},{r2},{r3},{r4}]')

env_structure = [r1,r2,r3,r4]
env = DDoSEnvironment(structure=env_structure,device=device)

epi_reward = 0

state = env.reset()
state = torch.from_numpy(state).float() # (num_routers, state_size)



for t in range(steps): # 각 에피소드 당 타임스텝
    # 모든 라우터(에이전트)에 대해 행동 선택
        
    actions = []
    for agent_idx in range(len(env.throttling_routers)):
        # 각 라우터는 자신의 상태를 기반으로 행동
        agent_state = state[agent_idx].unsqueeze(0)
        action = agent.select_action(agent_state)
        actions.append(action.item())

    # 환경에서 행동 수행 및 결과 수신
    next_state, reward, done = env.step(actions)
    reward = torch.tensor([reward], dtype=torch.float)
        #if t%10 == 0:
            #print(f"step{t}) reward : {reward}")
    epi_reward += reward
    next_state = torch.from_numpy(next_state).float()

    state = next_state

        
    if done:
        break

env.drawGraph(steps)