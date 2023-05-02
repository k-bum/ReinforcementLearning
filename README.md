# Reinforcement_Learning
2023-1 SWCON Reinforcement Learning lecture
# Reinforcement Learning

## 강화학습

- 시행착오를 통해 학습해 일련의 결정을 내리는 기계학습 알고리즘
<img width="756" alt="Untitled" src="https://user-images.githubusercontent.com/96854885/235691785-6aa69d27-b1d6-4ed7-8798-493eba097787.png">

Terminology

- Environment : Agent를 제외한 모든 요소 (agent가 취할 수 있는 행동, 그에 대한 보상 등)
- Agent : 학습하는 대상인 동시에 environment에서 action을 취하는 개체
- State : 현재 시점에서 상황이 어떤지 나타내는 값의 집합(가능한 모든 상태의 집합 : state space)
- Action : Agent가 취할 수 있는 선택지(가능한 모든 action의 집합 : action space)

→ Agent는 environment를 통해 state 정보를 받고, 그에 따라 action을 취한다.  state와 action에 따라 environment로부터 보상을 받고, action으로 인해 바뀐 state 정보를 전달 받아 다음 action으로 이어지는 순환과정이다. (sequential decision making problems)

→ 강화학습의 궁극적인 목표는 최대한 보상을 많이 받는 것이다. (Maximize expected return)

- Episode : Initial state부터 terminal state까지 일련의 과정
- Reward : 한 행동을 했을 때 받는 보상
- Return : Episode가 끝났을 때까지 받는 모든 보상의 합

## Grid world

<img width="756" alt="Untitled" src="https://user-images.githubusercontent.com/96854885/235692200-096f0154-43c0-4fb3-9f52-364d7767a4e7.png">

- Q-value : 어떤 action을 했을 때 미래에 받을 것이라고 예상되는 return의 값
- Greedy action : 그 순간에 최적이라고 생각되는 것을 선택해 나가는 방식으로 진행해 최종 상태에 도달

→ 더 좋은 path가 있음에도 불구하고 같은 path만 선택하게 된다.

→ Exploration  : 어느 정도는 greedy action을 따르지 않고 더 좋은 path를 찾기 위해 탐험해야 한다. 

- Epsilon-greedy action : epsilon 확률로 random action을 취하고(exploration), 1 - epsilon 확률로 greedy action을 취한다.(exploitation)
- Decay epsilon-greedy : 학습이 진행됨에 따라 epsilon의 값을 0에 가깝게 만든다.
- Discount factor : 0~1 사이의 gamma(hyper parameter)를 사용해 먼 시점에 받는 reward에 낮은 가중치를 두고 가까운 시점에 받는 reward에 높은 가중치를 준다.

→ Discount factor를 사용하면 더 효울적인 path를 찾을 수 있고, 현재의 reward와 미래의 reward 중 어떤 reward에 중점을 둘 지 선택하는 것이 가능하며, 수학적 편리성을 얻을 수 있다.(수렴이 더 잘 된다.)

## Markov process

### Markov property

과거 상태(s_1, s_2, … , s_t-1)와 현재 상태(s_t)가 주어졌을 때, 미래 상태(s_t+1)는 과거 상태와는 독립적으로 현재 상태에 의해서만 결정된다.

- The future is independent of the past given the present
- A state S_t is a Markov if and only if

$$
P[S_{t+1} | S_t] = P[S_{t+1} | S_1, S_2, … , S_t]
$$

### Markov Reward Process

- Reward function : 어떤 상태 s에 도달 했을 때 받게 되는 보상

$$
R = E[R_t |s_t = s]
$$

- Return function : t 시점으로 부터 미래에 받을 discounted reward의 합

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + ... = R_t + \gamma G_{t+1}
$$

### Markov Decision Process

- Transition probability : 현재 state에서 어떤 action을 취했을 때 다음 state로 갈 확률

$$
P^a_{SS'} = P[s_{t+1}|s_1,a_1,s_2,a_2,...,s_t,a_t] = P[s_{t+1} = s'|s_t=s,a_t=a] 
$$

$$
R_s^a = E[R_{t+1} | s_t=s, a_t=a]
$$

→ 강화학습의 최종 목표) Maximize the expected return → Finding an optimal policy

1) Prediction : Policy가 주어졌을 때 각 state의 value를 평가하는 문제

2) Control : Optimal policy를 찾는 문제

### Bellman Equation

<img width="756" alt="Untitled" src="https://user-images.githubusercontent.com/96854885/235692754-48545233-1d12-476d-8089-5c58c4e3fb76.png">

### State-value function

지금부터 기대되는 Return 

$$
v_{\pi}(s_t) = E[R_{t+1} + \gamma v(s_{t+1})|s_t = s] 
$$

<img width="300" height="100" alt="4" src="https://user-images.githubusercontent.com/96854885/235693565-bc7aae4c-6d0b-4384-9f6c-658544433a26.png">
<img width="400" height="80" alt="5" src="https://user-images.githubusercontent.com/96854885/235693604-8a60bfac-112d-4cc4-a79e-dbf8fe48a3f5.png">

### Action-value function

지금 행동부터 기대되는 Return 

$$
q_{\pi}(s_t, a_t) = E_{\pi}[R_t + \gamma q_{\pi}(s_{t+1}, a_{t+1})|s_t=s, a_t=a]
$$

<img width="300" height="100" alt="6" src="https://user-images.githubusercontent.com/96854885/235693710-cadbfef2-ef9b-4845-aa9d-ff7cf2458a8a.png">
<img width="400" height="80" alt="7" src="https://user-images.githubusercontent.com/96854885/235693739-84c425a2-e63b-483c-b3a5-4b47d0deecd0.png">


## Policy iteration

- Policy evaluation + Policy improvement
- Policy evaluation : Policy가 고정된 상태에서 value function을 학습하는 것
- Policy improvement : Value function을 고정한 상태에서 최적의 Policy를 선택하는 것

$$
v_{\pi}(s) = \Sigma\pi(a|s)(R_s^a+\gamma \Sigma P_{ss'}^av_{\pi}(s'))
$$

![Untitled](Reinforcement%20Learning%207a2097c8aa734c38848e5c164f83891e/Untitled%207.png)

### Value iteration

$$
v_*(s) = max_a(R_s^a+\gamma \Sigma P_{ss'}^av_{*}(s'))
$$

## Model-based vs Model-free

Model-base : MDP에 대한 정보를 알 때(transition probability, reward를 알고 있을 때)

Model-free : MDP에 대한 정보를 모를 때 → Monte-Carlo(MC) / Temporal Difference(TD)

## Monte-Carlo method vs Temporal Difference learning

Monte-carlo method : Episode가 끝나면 update

Temporal Difference learning : Episode가 끝나기 전에 update

<img width="756" alt="Untitled" src="https://user-images.githubusercontent.com/96854885/235693284-9d822db1-0da2-49d9-9707-dc6853089f1b.png">

## On-policy vs Off-policy

On-policy : Target policy와 behavior policy가 같은 경우(SARSA) → 안전한 path를 찾는다.

Off-policy : Target policy와 behavior policy가 다른 경우(Q-learning) 

![Untitled](Reinforcement%20Learning%207a2097c8aa734c38848e5c164f83891e/Untitled%209.png)
