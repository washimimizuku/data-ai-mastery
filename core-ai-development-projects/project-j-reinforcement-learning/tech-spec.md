# Tech Spec: Reinforcement Learning Agent

## Stack
Gymnasium, Stable-Baselines3, PyTorch

## Environments
- CartPole-v1
- LunarLander-v2
- Custom environment

## Algorithms
```python
from stable_baselines3 import DQN, PPO, A2C

# DQN
model = DQN("MlpPolicy", env, verbose=1)

# PPO
model = PPO("MlpPolicy", env, verbose=1)

# A2C
model = A2C("MlpPolicy", env, verbose=1)
```

## Training
```python
model.learn(total_timesteps=100000)
model.save("agent")
```

## Evaluation
```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
```
