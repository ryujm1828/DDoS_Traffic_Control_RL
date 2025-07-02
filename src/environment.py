import numpy as np
import random
import matplotlib.pyplot as plt

# --- 구성요소 클래스 ---

# Terminal : 트래픽을 생성하는 사용자(혹은 공격자)
class Terminal:
    def __init__(self, id, parent_router):
        self.id = id
        self.parent = parent_router

        self.profile_name = 'normal'
        self.behavior_params = {}

    #프로필 정의
    def set_profile(self, profile_name):
        self.profile_name = profile_name

        if profile_name == 'NormalUser':
            # 평균 트래픽은 낮지만, 때때로 튀는 트래픽이 있음
            self.behavior_params = {'rate': 0.4,'variability': 0.2,'burst_prob': 0.1, 'burst_scale': 5}
        elif profile_name == 'VideoUser':
            # 꾸준히 높은 트래픽을 유지
            self.behavior_params = {'rate': 2.0, 'variability': 0.2}
        elif profile_name == 'Attacker':
            # 매우 높은 트래픽을 변동 없이 계속 보냄
            self.behavior_params = {'rate': 10.0, 'life' : random.randint(20,30)}

    #트래픽 생성
    def generate_traffic(self):
        profile = self.profile_name
        params = self.behavior_params

        if profile == 'NormalUser':
            rate = np.random.normal(params['rate'], params['variability'])

            if random.random() < params['burst_prob']:
                rate *= params['burst_scale']
            return rate

        elif profile == 'VideoUser':
            return max(0, np.random.normal(params['rate'], params['variability']))

        elif profile == 'Attacker':
            return params['rate']

        return 0.0

# Router : 트래픽을 전달하는 장치
class Router:
    def __init__(self, id, router_type, parent=None):
        self.id = id
        self.router_type = router_type
        self.parent = parent
        self.children = []
        self.total_traffic = 0.0
        self.legitimate_traffic = 0.0

# --- 2. 메인 환경 클래스 ---
class DDoSEnvironment:
    def __init__(self, structure,device):
        """
        :param structure: [팀 수, 리더당 중간 라우터 수, 중간당 쓰로틀링 라우터 수, 쓰로틀링당 터미널 수]
                          예: [5, 2, 3, 2]
        """
        self.avg_throts = []
        self.rewards = []
        self.attack_steps = []
        
        self.structure_info = structure
        self.prob_legitimate = 0.55  #정상 유저 비율
        self.prob_attacker = 0.45    #공격자 비율
        self.legitimate_rate_range = [0, 1]  #정상 유저가 전송하는 트래픽 비율
        self.attack_rate_range = [2.5, 6]    #공격자가 전송하는 트래픽 비율

        # 네트워크 구성요소 저장용 리스트
        self.server = None
        self.team_leaders = []
        self.intermediate_routers = []
        self.throttling_routers = [] # 이들이 바로 강화학습 에이전트들
        self.terminals = []

        self._build_network()
        # 기반 서버 용량 상한(Us) 계산
        # Us = 터미널 개수 + delta(여유값)
        self.server_capacity_upper_bound = len(self.terminals) + 3

        self.time_step = 0
        self.attack_pro = 0.003 #각 확률별로 공격이 발생할 확률

        self.distributed_attack = True    #라우터 분산공격 여부
        self.num_attackers = 5      #공격자 수
        
        #사용자 분포
        self.profile_distribution = {
            'NormalUser': 0.80,
            'VideoUser': 0.20,
        }

    def numAttack(self):
        return len(self.attack_steps)
    
    def drawGraph(self,steps):
    
        fig, ax1 = plt.subplots()

        x = [i for i in range(1, steps + 1)]
        y1 = self.avg_throts  # 평균 쓰로틀링율
        y2 = self.rewards    # 리워드

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Average Throttling Ratio', color='blue') # 왼쪽 y 라벨
        p1 = ax1.plot(x, y1, linestyle="--", c='blue', label='avg_throttling_ratio')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0.0,1.0)
        
        # 공격 지점(scatter) 그리기
        x_attack = self.attack_steps
        y_attack = [0.5 for i in range(len(self.attack_steps))]
        p3 = ax1.scatter(x_attack, y_attack,
                        s=10,
                        c='red',
                        marker='o',
                        zorder=3,
                        label='attack')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Reward', color='pink') # 오른쪽 y 라벨
        ax2.set_ylim(-1.0,1.0)
        p2 = ax2.plot(x, y2, linestyle="-", c='pink', label='reward')
        ax2.tick_params(axis='y', labelcolor='pink')

        plots = p1 + p2 + [p3]
        labels = [p.get_label() for p in plots]
        ax1.legend(plots, labels, loc='upper left')

        plt.show()
        plt.clf()
        return
    
    def _build_network(self):
        # 서버 생성
        self.server = Router(id='server', router_type='server')

        num_teams, num_inter_per_leader, num_throt_per_inter, num_term_per_throt = self.structure_info

        # 계층적으로 네트워크 생성 (서버-라우터-라우터-라우터-터미널)
        throttling_router_id_counter = 0
        for i in range(num_teams):
            leader = Router(id=f'L_{i}', router_type='leader', parent=self.server)
            self.team_leaders.append(leader)
            self.server.children.append(leader)

            for j in range(num_inter_per_leader):
                inter = Router(id=f'I_{i}-{j}', router_type='intermediate', parent=leader)
                self.intermediate_routers.append(inter)
                leader.children.append(inter)

                for k in range(num_throt_per_inter):
                    throt = Router(id=throttling_router_id_counter, router_type='throttling', parent=inter)
                    self.throttling_routers.append(throt)
                    inter.children.append(throt)
                    throttling_router_id_counter += 1

                    for l in range(num_term_per_throt):
                        term = Terminal(id=f'T_{i}-{j}-{k}-{l}', parent_router=throt)
                        self.terminals.append(term)
                        throt.children.append(term)

    def add_attacker(self,throt_i,step,i):
        throt = self.throttling_routers[throt_i]
        term = Terminal(id=f'A{step}_{throt.parent.parent.id}-{throt.parent.id}-{throt.id}-{i}', parent_router=throt)
        term.set_profile("Attacker")
        self.terminals.append(term)
        throt.children.append(term)
        self.attack_steps.append(step)
        
    def reset(self):
        self.avg_throts = []
        self.rewards = []
        self.attack_steps = []
        
        self.time_step = 0
        # 1. 모든 라우터의 트래픽 초기화
        for router in [self.server] + self.team_leaders + self.intermediate_routers + self.throttling_routers:
            router.total_traffic = 0.0
            router.legitimate_traffic = 0.0

        # 2. 남아있는 공격자 제거
        attackers = []
        for term in self.terminals:
            if term.profile_name == 'Attacker':
                attackers.append(term)

        for att in attackers:
            self.terminals.remove(att)
            att.parent.children.remove(att)
            del att

        # 3. 터미널 역할 및 트래픽 재할당
        profiles = list(self.profile_distribution.keys())
        probs = list(self.profile_distribution.values())
        
        for term in self.terminals:
            # 프로필을 랜덤하게 선택하여 할당
            assigned_profile = np.random.choice(profiles, p=probs)
            term.set_profile(assigned_profile)
        
        
            
        return self.get_state()

    def step(self, actions):
        self.time_step += 1
        sum_action = 0.0

        # 쓰로틀링 라우터 -> 중간 라우터 -> 팀 리더 -> 서버 순서로 트래픽 계산

        # 쓰로틀링 라우터 레벨
        total_legit_traffic = 0

        for i, throt_router in enumerate(self.throttling_routers):
            action = actions[i] * 0.1
            in_total_traffic = 0
            in_legit_traffic = 0
            for term in throt_router.children:
                # 고정된 traffic_rate 대신, 동적으로 트래픽 생성
                current_traffic = term.generate_traffic()
                in_total_traffic += current_traffic
                
                # Attacker일 경우 정상적인 트래픽으로 간주X
                if "Attacker" != term.profile_name:
                    in_legit_traffic += current_traffic
                else:   
                    #Attacker일 경우 수명 감소
                    term.behavior_params["life"] -= 1
                    if term.behavior_params["life"] <= 0:
                        self.terminals.remove(term)
                        term.parent.children.remove(term)
                        del term
                        
            # 쓰로틀링 적용
            throt_router.total_traffic = in_total_traffic * (1.0 - action)
            throt_router.legitimate_traffic = in_legit_traffic * (1.0 - action)
            
            sum_action += action
            
            total_legit_traffic += in_legit_traffic
        
        avg_action = sum_action/len(self.throttling_routers)
        self.avg_throts.append(avg_action)
        
        # 중간 라우터 레벨
        for inter_router in self.intermediate_routers:
            inter_router.total_traffic = sum(c.total_traffic for c in inter_router.children)
            inter_router.legitimate_traffic = sum(c.legitimate_traffic for c in inter_router.children)

        # 팀 리더 레벨
        for leader_router in self.team_leaders:
            leader_router.total_traffic = sum(c.total_traffic for c in leader_router.children)
            leader_router.legitimate_traffic = sum(c.legitimate_traffic for c in leader_router.children)

        # 서버 레벨
        self.server.total_traffic = sum(c.total_traffic for c in self.server.children)
        self.server.legitimate_traffic = sum(c.legitimate_traffic for c in self.server.children)

        # 2. 보상 계산
        reward = 0
        server_overloaded = self.server.total_traffic > self.server_capacity_upper_bound

        for leader in self.team_leaders:
            # 팀 리더가 과부하이고 서버 전체도 과부하이면 페널티
            hypothetical_bound = self.server_capacity_upper_bound / len(self.team_leaders)
            if leader.total_traffic > hypothetical_bound and server_overloaded:
                reward = -1.0
                break

        if reward == 0:
            # 통과된 전체 합법 트래픽 / 원래 발생한 전체 합법 트래픽
            if in_legit_traffic > 0:
                reward = self.server.legitimate_traffic / total_legit_traffic
            else:
                reward = 1.0 # 합법 트래픽이 없으면 보상 1

        self.rewards.append(reward)
        
        #일정 확률로 공격자 추가
        if random.random() <= self.attack_pro:
            if self.distributed_attack == True:
                for i in range(self.num_attackers):
                    throt_i = random.randint(0,len(self.throttling_routers)-1) #임의의 쓰로틀링 라우터 선택
                    self.add_attacker(throt_i,self.time_step,i)
            else:
                throt_i = random.randint(0,len(self.throttling_routers)-1) #임의의 쓰로틀링 라우터 선택
                for i in range(self.num_attackers):
                    self.add_attacker(throt_i,self.time_step,i)
            
        # 3. 다음 상태 가져오기
        next_state = self.get_state()
        done = False
        
        return next_state, reward, done

    def get_state(self):
        # 각 쓰로틀링 라우터(에이전트)에 대한 상태 벡터를 생성
        states = []
        for agent_router in self.throttling_routers:
            # 상태: [서버 트래픽, 팀리더 트래픽, 중간라우터 트래픽, 자신의 트래픽,팀리더 개수, 중간라우터 개수, 쓰로틀링 라우터 개수, 터미널 개수]
            r_s = self.server.total_traffic
            r_1 = agent_router.parent.parent.total_traffic # 팀 리더
            r_2 = agent_router.parent.total_traffic # 중간 라우터
            r_3 = agent_router.total_traffic # 자신
            
            state_vec = [r_s, r_1, r_2, r_3]
            state_vec += self.structure_info    #라우터와 터미널 개수 추가

            states.append(state_vec)

        return np.array(states, dtype=np.float32)
