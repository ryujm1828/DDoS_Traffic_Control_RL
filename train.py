import torch
import random
import matplotlib.pyplot as plt
from src import DDoSEnvironment, DQNAgent

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"현재 사용 중인 장치: {device}")


env_structure = [5, 2, 3, 3]
env = DDoSEnvironment(structure=env_structure,device=device)
state_size = 8
action_size = 10 # 스로틀링 레벨 0.0 ~ 0.9

is_env_change = False

# 모든 라우터가 하나의 신경망을 공유 (Parameter Sharing)
agent = DQNAgent(state_size, action_size,device)
agent.policy_net.to(device)
agent.target_net.to(device)

num_episodes = 10001 # 학습 에피소드 수
steps = 200
train_inteval = 8

sum_rewards = []

for i_episode in range(num_episodes):
    epi_reward = 0
    print(f"{i_episode}번째 학습")
        
    if is_env_change == True:
        del env
        r1 = random.randint(2,6)
        r2 = random.randint(2,6)
        r3 = random.randint(2,6)
        r4 = random.randint(2,6)
        env_structure = [r1,r2,r3,r4]
        env = DDoSEnvironment(structure=env_structure,device=device)
    
    state = env.reset()
    state = torch.from_numpy(state).float() # (num_routers, state_size)
    
    for t in range(steps): # 각 에피소드 당 타임스텝
        # 모든 라우터(에이전트)에 대해 행동 선택
        actions = []
        for agent_idx in range(len(env.throttling_routers)):
            # 각 라우터는 자신의 상태를 기반으로 행동
            agent_state = state[agent_idx].unsqueeze(0)
            action = agent.select_action(agent_state).to(device)
            actions.append(action.item())

        # 환경에서 행동 수행 및 결과 수신
        next_state, reward, done = env.step(actions)
        reward = torch.tensor([reward], dtype=torch.float)
        #if t%10 == 0:
            #print(f"step{t}) reward : {reward}")
        epi_reward += reward
        next_state = torch.from_numpy(next_state).float()

        # 경험을 메모리에 저장 (모든 에이전트에 대해)
        for agent_idx in range(len(env.throttling_routers)):
            agent.memory.push(state[agent_idx].unsqueeze(0),
                              torch.tensor([[actions[agent_idx]]]),
                              reward,
                              next_state[agent_idx].unsqueeze(0))

        state = next_state
        
        agent.learn()   # agent 학습
        
        if done:
            break
        
    if i_episode % 100 == 0: # 주기적으로 타겟 네트워크 업데이트
        agent.update_target_net()
    
    if i_episode % 500 == 0:
        torch.save(agent.policy_net.state_dict(), f'./results/model_epi{i_episode}.pth')

    sum_rewards.append(epi_reward)

x = [i for i in range(1,num_episodes+1)]
y = sum_rewards    #에피소드의 총 보상 합
plt.plot(x,y,linestyle="-",c='pink',label = 'sum of reward')
plt.legend()
plt.show()
plt.clf()

env.drawGraph(steps)

torch.save(agent.policy_net.state_dict(), f'./results/model_epi{num_episodes}.pth')