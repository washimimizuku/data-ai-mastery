# Project J: Reinforcement Learning Agent

## Objective

Build comprehensive reinforcement learning agents using multiple algorithms (DQN, PPO, A2C, SAC) to solve classic control and continuous action environments, demonstrating training, evaluation, hyperparameter tuning, and deployment of RL policies.

**What You'll Build**: A complete RL training pipeline with multiple algorithms, environment wrappers, training visualization, policy evaluation, hyperparameter optimization, custom environments, and deployment-ready agents that can solve CartPole, LunarLander, and custom tasks.

**What You'll Learn**: RL fundamentals (Q-learning, policy gradients), deep RL algorithms (DQN, PPO, A2C, SAC), environment design, reward shaping, exploration strategies, training stability, policy evaluation, hyperparameter tuning, and production deployment of RL agents.

## Time Estimate

**2 days (16 hours)** - Following the implementation plan

### Day 1 (8 hours)
- **Hour 1**: Setup & basics (install Gymnasium, test environments, understand RL concepts)
- **Hours 2-3**: DQN implementation (CartPole, training loop, Q-network, replay buffer)
- **Hours 4-5**: PPO implementation (policy gradients, advantage estimation, clipping)
- **Hours 6-7**: Environment exploration (LunarLander, continuous actions, custom envs)
- **Hour 8**: Training visualization (rewards, losses, episode lengths)

### Day 2 (8 hours)
- **Hours 1-2**: Advanced algorithms (A2C, SAC for continuous control)
- **Hours 3-4**: Hyperparameter tuning (Optuna integration, grid search)
- **Hours 5-6**: Policy evaluation (success rate, robustness testing, visualization)
- **Hour 7**: Custom environment (design, implement, train agent)
- **Hour 8**: Deployment & documentation (save/load models, inference API)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 51-70
  - Days 51-55: RL fundamentals (MDP, Q-learning, policy gradients)
  - Days 56-60: Deep RL (DQN, PPO, actor-critic)
  - Days 61-65: Advanced RL (exploration, reward shaping)
  - Days 66-70: RL applications and deployment
- [30 Days of Python](https://github.com/washimimizuku/30-days-python-data-ai) - Days 1-20

### Technical Requirements
- Python 3.11+ installed
- 8GB+ RAM
- GPU recommended for faster training (optional)
- Understanding of neural networks
- Basic knowledge of RL concepts (states, actions, rewards)

### Tools Needed
- Python with gymnasium, stable-baselines3, torch
- Matplotlib for visualization
- Optuna for hyperparameter tuning
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install gymnasium[classic-control,box2d]
pip install stable-baselines3[extra]
pip install torch torchvision

# Install visualization and tuning
pip install matplotlib tensorboard optuna

# Install utilities
pip install tqdm pandas

# Verify installation
python -c "import gymnasium as gym; print('✓ Gymnasium installed')"
python -c "from stable_baselines3 import PPO; print('✓ Stable-Baselines3 installed')"
```

### Step 3: Test Basic Environment
```python
# test_env.py
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset environment
observation, info = env.reset(seed=42)
print(f"Initial observation: {observation}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Run random agent
episode_reward = 0
for _ in range(200):
    # Random action
    action = env.action_space.sample()
    
    # Step environment
    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    
    if terminated or truncated:
        print(f"Episode finished with reward: {episode_reward}")
        break

env.close()
```


### Step 4: Train DQN Agent
```python
# src/agents/dqn_agent.py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

class DQNTrainer:
    """Train DQN agent on discrete action environments"""
    
    def __init__(self, env_name: str = "CartPole-v1"):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.eval_env = gym.make(env_name)
        self.model = None
    
    def create_model(self, learning_rate: float = 1e-3, buffer_size: int = 50000):
        """Create DQN model"""
        
        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log="./logs/dqn/"
        )
        
        print(f"✓ DQN model created for {self.env_name}")
        return self.model
    
    def train(self, total_timesteps: int = 100000, save_path: str = "./models/dqn"):
        """Train DQN agent"""
        
        if self.model is None:
            self.create_model()
        
        # Create callbacks
        os.makedirs(save_path, exist_ok=True)
        
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=save_path,
            name_prefix="dqn_checkpoint"
        )
        
        # Train
        print(f"Training DQN for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model
        self.model.save(f"{save_path}/dqn_final")
        print(f"✓ Model saved to {save_path}")
    
    def evaluate(self, n_episodes: int = 10):
        """Evaluate trained agent"""
        from stable_baselines3.common.evaluation import evaluate_policy
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True
        )
        
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward
    
    def visualize(self, n_episodes: int = 3):
        """Visualize agent performance"""
        env = gym.make(self.env_name, render_mode="human")
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
        
        env.close()

# Usage
if __name__ == "__main__":
    trainer = DQNTrainer("CartPole-v1")
    trainer.create_model()
    trainer.train(total_timesteps=50000)
    trainer.evaluate()
    trainer.visualize()
```


### Step 5: Train PPO Agent
```python
# src/agents/ppo_agent.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

class PPOTrainer:
    """Train PPO agent (works for both discrete and continuous)"""
    
    def __init__(self, env_name: str = "LunarLander-v2"):
        self.env_name = env_name
        self.env = self._make_env()
        self.eval_env = self._make_env()
        self.model = None
    
    def _make_env(self):
        """Create and wrap environment"""
        env = gym.make(self.env_name)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env
    
    def create_model(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64
    ):
        """Create PPO model"""
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./logs/ppo/"
        )
        
        print(f"✓ PPO model created for {self.env_name}")
        return self.model
    
    def train(self, total_timesteps: int = 500000, save_path: str = "./models/ppo"):
        """Train PPO agent"""
        
        if self.model is None:
            self.create_model()
        
        # Create callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=10000,
            deterministic=True
        )
        
        # Train
        print(f"Training PPO for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save
        self.model.save(f"{save_path}/ppo_final")
        self.env.save(f"{save_path}/vec_normalize.pkl")
        print(f"✓ Model saved to {save_path}")
    
    def load(self, model_path: str):
        """Load trained model"""
        self.model = PPO.load(model_path, env=self.env)
        self.env = VecNormalize.load(f"{model_path}/vec_normalize.pkl", self.env)
        print(f"✓ Model loaded from {model_path}")

# Usage
if __name__ == "__main__":
    trainer = PPOTrainer("LunarLander-v2")
    trainer.create_model()
    trainer.train(total_timesteps=500000)
```


### Step 6: Implement Training Visualization
```python
# src/utils/visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

class TrainingVisualizer:
    """Visualize RL training progress"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
    
    def plot_training_progress(self, algorithm: str = "ppo"):
        """Plot reward progression during training"""
        
        # Load tensorboard logs
        from tensorboard.backend.event_processing import event_accumulator
        
        log_path = self.log_dir / algorithm
        ea = event_accumulator.EventAccumulator(str(log_path))
        ea.Reload()
        
        # Extract rewards
        rewards = ea.Scalars('rollout/ep_rew_mean')
        steps = [r.step for r in rewards]
        values = [r.value for r in rewards]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(steps, values, label='Episode Reward', alpha=0.6)
        
        # Add moving average
        window = 10
        if len(values) >= window:
            moving_avg = pd.Series(values).rolling(window=window).mean()
            plt.plot(steps, moving_avg, label=f'Moving Average ({window})', linewidth=2)
        
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Episode Reward')
        plt.title(f'{algorithm.upper()} Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{algorithm}_training_progress.png', dpi=300)
        print(f"✓ Plot saved to {algorithm}_training_progress.png")
    
    def compare_algorithms(self, algorithms: list):
        """Compare multiple algorithms"""
        
        plt.figure(figsize=(14, 7))
        
        for algo in algorithms:
            log_path = self.log_dir / algo
            if not log_path.exists():
                continue
            
            ea = event_accumulator.EventAccumulator(str(log_path))
            ea.Reload()
            
            rewards = ea.Scalars('rollout/ep_rew_mean')
            steps = [r.step for r in rewards]
            values = [r.value for r in rewards]
            
            plt.plot(steps, values, label=algo.upper(), alpha=0.7)
        
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Episode Reward')
        plt.title('Algorithm Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300)
        print("✓ Comparison plot saved")
    
    def plot_episode_lengths(self, algorithm: str = "ppo"):
        """Plot episode length progression"""
        
        from tensorboard.backend.event_processing import event_accumulator
        
        log_path = self.log_dir / algorithm
        ea = event_accumulator.EventAccumulator(str(log_path))
        ea.Reload()
        
        lengths = ea.Scalars('rollout/ep_len_mean')
        steps = [l.step for l in lengths]
        values = [l.value for l in lengths]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, values)
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Episode Length')
        plt.title(f'{algorithm.upper()} Episode Length')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{algorithm}_episode_length.png', dpi=300)

# Usage
if __name__ == "__main__":
    viz = TrainingVisualizer()
    viz.plot_training_progress("ppo")
    viz.compare_algorithms(["dqn", "ppo", "a2c"])
```


### Step 7: Hyperparameter Tuning with Optuna
```python
# src/tuning/hyperparameter_tuning.py
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

class HyperparameterTuner:
    """Tune RL hyperparameters with Optuna"""
    
    def __init__(self, env_name: str = "CartPole-v1", n_trials: int = 50):
        self.env_name = env_name
        self.n_trials = n_trials
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna"""
        
        # Sample hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        n_epochs = trial.suggest_int("n_epochs", 3, 30)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
        
        # Create environment
        env = gym.make(self.env_name)
        
        # Create model with sampled hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=0
        )
        
        # Train
        model.learn(total_timesteps=50000)
        
        # Evaluate
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        
        env.close()
        
        return mean_reward
    
    def tune(self):
        """Run hyperparameter tuning"""
        
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10000)
        )
        
        print(f"Starting hyperparameter tuning ({self.n_trials} trials)...")
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print("\n✓ Tuning complete!")
        print(f"Best reward: {study.best_value:.2f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        return study.best_params

# Usage
if __name__ == "__main__":
    tuner = HyperparameterTuner("CartPole-v1", n_trials=20)
    best_params = tuner.tune()
```


### Step 8: Create Custom Environment
```python
# src/envs/custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleGridWorld(gym.Env):
    """Custom grid world environment"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, size: int = 5, render_mode=None):
        super().__init__()
        
        self.size = size
        self.render_mode = render_mode
        
        # Define action and observation space
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observations: agent position (x, y)
        self.observation_space = spaces.Box(
            low=0,
            high=size-1,
            shape=(2,),
            dtype=np.float32
        )
        
        # Goal position
        self.goal_pos = np.array([size-1, size-1])
        
        # Agent position
        self.agent_pos = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start at random position
        self.agent_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        # Move agent
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # right
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # down
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 3:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        
        # Check if goal reached
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        
        # Calculate reward
        if terminated:
            reward = 10.0
        else:
            # Negative reward based on distance to goal
            distance = np.linalg.norm(self.agent_pos - self.goal_pos)
            reward = -distance * 0.1
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _get_obs(self):
        return self.agent_pos.astype(np.float32)
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(self.agent_pos - self.goal_pos)
        }
    
    def render(self):
        if self.render_mode == "human":
            # Simple text rendering
            grid = np.zeros((self.size, self.size), dtype=str)
            grid[:] = '.'
            grid[self.agent_pos[1], self.agent_pos[0]] = 'A'
            grid[self.goal_pos[1], self.goal_pos[0]] = 'G'
            
            print("\n" + "\n".join([" ".join(row) for row in grid]))

# Register custom environment
from gymnasium.envs.registration import register

register(
    id='SimpleGridWorld-v0',
    entry_point='src.envs.custom_env:SimpleGridWorld',
    max_episode_steps=100,
)

# Usage
if __name__ == "__main__":
    env = gym.make('SimpleGridWorld-v0', render_mode="human")
    
    obs, info = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated:
            print("Goal reached!")
            break
```

## Key Features to Implement

### 1. RL Algorithms
- **DQN**: Deep Q-Network for discrete actions
- **PPO**: Proximal Policy Optimization (discrete & continuous)
- **A2C**: Advantage Actor-Critic
- **SAC**: Soft Actor-Critic (continuous actions)
- **TD3**: Twin Delayed DDPG (continuous actions)

### 2. Environments
- **CartPole-v1**: Balance pole on cart
- **LunarLander-v2**: Land spacecraft safely
- **MountainCar-v0**: Drive car up hill
- **Pendulum-v1**: Swing pendulum upright
- **Custom environments**: Design your own

### 3. Training Features
- **Replay buffer**: Experience replay for DQN
- **Vectorized environments**: Parallel training
- **Callbacks**: Evaluation, checkpointing, early stopping
- **Tensorboard logging**: Track training metrics
- **Progress bars**: Monitor training progress

### 4. Evaluation
- **Policy evaluation**: Mean reward over episodes
- **Success rate**: Percentage of successful episodes
- **Robustness testing**: Performance under noise
- **Visualization**: Watch agent perform

### 5. Hyperparameter Tuning
- **Optuna integration**: Automated tuning
- **Grid search**: Systematic exploration
- **Random search**: Quick baseline
- **Best practices**: Learning rate, batch size, etc.

### 6. Deployment
- **Model saving/loading**: Persist trained agents
- **Inference API**: FastAPI endpoint
- **Web demo**: Gradio interface
- **Docker**: Containerized deployment


## Success Criteria

By the end of this project, you should have:

- [ ] DQN agent solving CartPole (reward > 195)
- [ ] PPO agent solving LunarLander (reward > 200)
- [ ] A2C agent trained on continuous environment
- [ ] Training visualization with reward curves
- [ ] Hyperparameter tuning completed
- [ ] Custom environment created and solved
- [ ] Policy evaluation metrics calculated
- [ ] Model saving/loading working
- [ ] Inference API deployed
- [ ] Gradio demo functional
- [ ] Performance benchmarks documented
- [ ] GitHub repository with examples

## Learning Outcomes

After completing this project, you'll be able to:

- Understand RL fundamentals (MDP, Q-learning, policy gradients)
- Implement and train DQN agents
- Use PPO for both discrete and continuous actions
- Design custom RL environments
- Shape rewards for better learning
- Tune hyperparameters effectively
- Evaluate RL policies rigorously
- Visualize training progress
- Deploy RL agents to production
- Debug training instabilities
- Compare different RL algorithms
- Explain exploration vs exploitation

## Expected Performance

Based on typical results:

**CartPole-v1** (DQN):
- Training time: 5-10 minutes (50K timesteps)
- Success threshold: 195 reward
- Typical final reward: 200+ (perfect)

**LunarLander-v2** (PPO):
- Training time: 30-60 minutes (500K timesteps)
- Success threshold: 200 reward
- Typical final reward: 250+

**Custom GridWorld** (PPO):
- Training time: 5-10 minutes (100K timesteps)
- Success rate: 90%+
- Convergence: 50K timesteps

## Project Structure

```
project-j-reinforcement-learning/
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py
│   │   ├── ppo_agent.py
│   │   ├── a2c_agent.py
│   │   └── sac_agent.py
│   ├── envs/
│   │   ├── custom_env.py
│   │   └── wrappers.py
│   ├── tuning/
│   │   ├── hyperparameter_tuning.py
│   │   └── grid_search.py
│   ├── utils/
│   │   ├── visualization.py
│   │   ├── evaluation.py
│   │   └── callbacks.py
│   ├── api/
│   │   └── inference_api.py
│   └── ui/
│       └── gradio_demo.py
├── models/
│   ├── dqn/
│   ├── ppo/
│   └── custom/
├── logs/
│   ├── dqn/
│   └── ppo/
├── notebooks/
│   ├── 01_dqn_cartpole.ipynb
│   ├── 02_ppo_lunarlander.ipynb
│   ├── 03_custom_environment.ipynb
│   └── 04_hyperparameter_tuning.ipynb
├── tests/
│   └── test_agents.py
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: Training Instability
**Problem**: Reward fluctuates wildly, doesn't converge
**Solution**: Normalize observations, tune learning rate, use reward clipping
```python
from stable_baselines3.common.vec_env import VecNormalize

# Normalize observations and rewards
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

# Lower learning rate
model = PPO("MlpPolicy", env, learning_rate=1e-4)
```

### Challenge 2: Slow Learning
**Problem**: Agent takes too long to learn
**Solution**: Increase batch size, use parallel environments, tune exploration
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Parallel environments
def make_env():
    return gym.make("CartPole-v1")

env = SubprocVecEnv([make_env for _ in range(4)])

# Larger batch size
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=256)
```

### Challenge 3: Agent Gets Stuck
**Problem**: Agent learns suboptimal policy
**Solution**: Increase exploration, shape rewards, try different algorithm
```python
# Increase entropy coefficient for more exploration
model = PPO("MlpPolicy", env, ent_coef=0.01)

# Shape rewards to guide learning
def shaped_reward(reward, info):
    # Add bonus for getting closer to goal
    distance_bonus = -info['distance'] * 0.1
    return reward + distance_bonus
```

### Challenge 4: Overfitting to Training
**Problem**: Agent performs well in training but poorly in evaluation
**Solution**: Use separate eval environment, add noise, regularization
```python
# Separate evaluation environment
eval_env = gym.make("CartPole-v1")

# Evaluate regularly
eval_callback = EvalCallback(
    eval_env,
    eval_freq=5000,
    n_eval_episodes=20,
    deterministic=True
)

model.learn(total_timesteps=100000, callback=eval_callback)
```

## Advanced Techniques

### 1. Curriculum Learning
```python
# Start with easier task, gradually increase difficulty
class CurriculumEnv(gym.Wrapper):
    def __init__(self, env, difficulty=0.5):
        super().__init__(env)
        self.difficulty = difficulty
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Scale reward based on difficulty
        reward *= self.difficulty
        return obs, reward, done, truncated, info
    
    def increase_difficulty(self):
        self.difficulty = min(1.0, self.difficulty + 0.1)

# Train with increasing difficulty
env = CurriculumEnv(gym.make("LunarLander-v2"))
for i in range(10):
    model.learn(total_timesteps=50000)
    env.increase_difficulty()
```

### 2. Reward Shaping
```python
# Add intermediate rewards to guide learning
class RewardShapingWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Add shaping rewards
        if 'distance_to_goal' in info:
            # Reward for getting closer
            shaped_reward = reward - 0.01 * info['distance_to_goal']
        else:
            shaped_reward = reward
        
        return obs, shaped_reward, done, truncated, info
```

### 3. Multi-Agent RL
```python
# Train multiple agents simultaneously
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank):
    def _init():
        env = gym.make("CartPole-v1")
        env.reset(seed=rank)
        return env
    return _init

# Create 8 parallel environments
env = SubprocVecEnv([make_env(i) for i in range(8)])

# Train with parallel experience collection
model = PPO("MlpPolicy", env, n_steps=128)
model.learn(total_timesteps=100000)
```

### 4. Transfer Learning
```python
# Transfer knowledge from one task to another
# Train on CartPole
model_cartpole = PPO("MlpPolicy", gym.make("CartPole-v1"))
model_cartpole.learn(total_timesteps=50000)

# Transfer to similar task
model_acrobot = PPO("MlpPolicy", gym.make("Acrobot-v1"))

# Copy policy network weights
model_acrobot.policy.load_state_dict(
    model_cartpole.policy.state_dict(),
    strict=False  # Allow partial loading
)

# Fine-tune on new task
model_acrobot.learn(total_timesteps=50000)
```


## Troubleshooting

### Installation Issues

**Problem**: `gymnasium[box2d]` fails to install
```bash
# Solution: Install Box2D separately
# On macOS
brew install swig
pip install box2d-py

# On Ubuntu
sudo apt-get install swig
pip install box2d-py

# Then install gymnasium
pip install gymnasium
```

**Problem**: CUDA/GPU not detected
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: `ModuleNotFoundError: No module named 'stable_baselines3'`
```bash
# Install with all extras
pip install stable-baselines3[extra]

# Or minimal install
pip install stable-baselines3
```

### Runtime Errors

**Problem**: `ValueError: could not broadcast input array`
```python
# Solution: Check observation space matches
print(f"Observation space: {env.observation_space}")
print(f"Observation shape: {obs.shape}")

# Ensure observation is correct shape
obs = obs.reshape(env.observation_space.shape)
```

**Problem**: `AssertionError: The algorithm only supports (<class 'gym.spaces.box.Box'>,) as action spaces`
```python
# Solution: Use correct algorithm for action space
# Discrete actions: DQN, A2C, PPO
# Continuous actions: SAC, TD3, PPO

# Check action space
print(f"Action space: {env.action_space}")

# Use appropriate algorithm
if isinstance(env.action_space, gym.spaces.Discrete):
    model = PPO("MlpPolicy", env)  # or DQN, A2C
else:
    model = SAC("MlpPolicy", env)  # or TD3
```

**Problem**: Training crashes with `NaN` values
```python
# Solution: Add gradient clipping and normalize
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

model = PPO(
    "MlpPolicy",
    env,
    max_grad_norm=0.5,  # Gradient clipping
    learning_rate=3e-4,
    verbose=1
)
```

### Performance Issues

**Problem**: Training is very slow
```python
# Solution 1: Use parallel environments
from stable_baselines3.common.vec_env import SubprocVecEnv

env = SubprocVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])

# Solution 2: Reduce logging frequency
model = PPO("MlpPolicy", env, verbose=0)

# Solution 3: Use GPU
model = PPO("MlpPolicy", env, device="cuda")
```

**Problem**: High memory usage
```python
# Solution: Reduce buffer size and batch size
model = DQN(
    "MlpPolicy",
    env,
    buffer_size=10000,  # Smaller buffer
    batch_size=32,      # Smaller batches
)
```

**Problem**: Agent not learning
```python
# Debug checklist:
# 1. Check rewards are non-zero
print(f"Reward range: {env.reward_range}")

# 2. Verify environment is solvable
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")

# 3. Try different hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-3,  # Higher learning rate
    ent_coef=0.01,       # More exploration
    n_steps=2048,        # More steps per update
)

# 4. Check tensorboard logs
# tensorboard --logdir ./logs
```

### Environment Issues

**Problem**: Custom environment not working
```python
# Solution: Validate environment
from stable_baselines3.common.env_checker import check_env

env = SimpleGridWorld()
check_env(env)  # Will raise errors if invalid

# Common fixes:
# 1. Ensure reset() returns (obs, info)
# 2. Ensure step() returns (obs, reward, terminated, truncated, info)
# 3. Ensure observation matches observation_space
# 4. Ensure action is valid for action_space
```

**Problem**: Rendering doesn't work
```python
# Solution: Specify render mode
env = gym.make("CartPole-v1", render_mode="human")

# For custom environments
class MyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
    
    def render(self):
        if self.render_mode == "human":
            # Display to screen
            pass
        elif self.render_mode == "rgb_array":
            # Return numpy array
            return np.zeros((400, 600, 3), dtype=np.uint8)
```


## Production Deployment

### FastAPI Inference Server
```python
# src/api/inference_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

app = FastAPI(title="RL Agent API")

# Load model at startup
model = None
env = None

@app.on_event("startup")
async def load_model():
    global model, env
    env = gym.make("CartPole-v1")
    model = PPO.load("./models/ppo/ppo_final", env=env)
    print("✓ Model loaded")

class ObservationInput(BaseModel):
    observation: list[float]

class ActionOutput(BaseModel):
    action: int
    confidence: float

@app.post("/predict", response_model=ActionOutput)
async def predict(input_data: ObservationInput):
    """Predict action from observation"""
    try:
        obs = np.array(input_data.observation, dtype=np.float32)
        action, _states = model.predict(obs, deterministic=True)
        
        return ActionOutput(
            action=int(action),
            confidence=1.0  # Add actual confidence if available
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

# Run with: uvicorn src.api.inference_api:app --reload
```

### Gradio Demo
```python
# src/ui/gradio_demo.py
import gradio as gr
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
from PIL import Image

class RLDemo:
    def __init__(self, model_path: str, env_name: str):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.model = PPO.load(model_path, env=self.env)
    
    def run_episode(self):
        """Run one episode and return frames"""
        frames = []
        obs, info = self.env.reset()
        episode_reward = 0
        
        for _ in range(500):
            # Render frame
            frame = self.env.render()
            frames.append(frame)
            
            # Predict action
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        return frames, episode_reward
    
    def create_video(self):
        """Create video from episode"""
        frames, reward = self.run_episode()
        
        # Create animation
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
        
        im = ax.imshow(frames[0])
        
        def update(frame_idx):
            im.set_array(frames[frame_idx])
            return [im]
        
        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=50,
            blit=True
        )
        
        # Save to buffer
        buf = io.BytesIO()
        anim.save(buf, format='gif', writer='pillow')
        buf.seek(0)
        
        plt.close()
        
        return Image.open(buf), f"Episode Reward: {reward:.2f}"

# Create demo
demo_app = RLDemo("./models/ppo/ppo_final", "CartPole-v1")

def run_demo():
    video, reward = demo_app.create_video()
    return video, reward

# Gradio interface
with gr.Blocks(title="RL Agent Demo") as demo:
    gr.Markdown("# Reinforcement Learning Agent Demo")
    gr.Markdown("Watch a trained PPO agent solve CartPole-v1")
    
    with gr.Row():
        run_btn = gr.Button("Run Episode", variant="primary")
    
    with gr.Row():
        video_output = gr.Image(label="Episode Video")
        reward_output = gr.Textbox(label="Episode Reward")
    
    run_btn.click(
        fn=run_demo,
        outputs=[video_output, reward_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    swig \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rl-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/ppo/ppo_final
    restart: unless-stopped
  
  rl-demo:
    build: .
    command: python src/ui/gradio_demo.py
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

### Model Serving Best Practices
```python
# src/api/production_api.py
from fastapi import FastAPI, BackgroundTasks
from prometheus_client import Counter, Histogram, generate_latest
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

app = FastAPI()

@app.middleware("http")
async def add_metrics(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    prediction_duration.observe(duration)
    prediction_counter.inc()
    
    return response

@app.get("/metrics")
async def metrics():
    return generate_latest()

# Add health checks, rate limiting, caching, etc.
```


## Resources

### Documentation
- [Gymnasium Documentation](https://gymnasium.farama.org/) - Environment library
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [OpenAI Spinning Up](https://spinningup.openai.com/) - RL fundamentals
- [Hugging Face RL Course](https://huggingface.co/learn/deep-rl-course/) - Free course

### Papers
- [DQN Paper](https://arxiv.org/abs/1312.5602) - Playing Atari with Deep RL
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [SAC Paper](https://arxiv.org/abs/1801.01290) - Soft Actor-Critic
- [Rainbow DQN](https://arxiv.org/abs/1710.02298) - Combining improvements

### Tutorials
- [Stable-Baselines3 Tutorial](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)
- [Gymnasium Tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/)
- [Custom Environments](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Hyperparameter Tuning](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)

### Videos
- [DeepMind RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)
- [David Silver's RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- [Stable-Baselines3 Tutorials](https://www.youtube.com/c/MachineLearningwithPhil)

### Books
- "Reinforcement Learning: An Introduction" by Sutton & Barto
- "Deep Reinforcement Learning Hands-On" by Maxim Lapan
- "Grokking Deep Reinforcement Learning" by Miguel Morales

### Communities
- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)
- [Stable-Baselines3 Discord](https://discord.com/invite/aBwKsh7)
- [Hugging Face Discord](https://discord.gg/hugging-face)

### Tools
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualization
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) - Scalable RL


## Questions?

If you get stuck:

1. Check the troubleshooting section above
2. Review the implementation plan (`implementation-plan.md`)
3. Look at example notebooks in `notebooks/`
4. Check Stable-Baselines3 documentation
5. Ask in the project discussions
6. Review similar projects in the bootcamp

Common questions:
- **Which algorithm should I use?** Start with PPO (works for both discrete and continuous)
- **How long should I train?** Until reward plateaus (check tensorboard)
- **Why isn't my agent learning?** Check rewards, normalize observations, tune learning rate
- **How do I speed up training?** Use parallel environments, GPU, reduce logging
- **Can I use pre-trained models?** Yes, check Hugging Face Hub for RL models


## Related Projects

Continue your learning with these related projects:

**Core AI Development**:
- [Project A: Local RAG System](../project-a-local-rag/) - Document Q&A
- [Project B: LLM Fine-tuning](../project-b-llm-finetuning/) - Custom models
- [Project C: Multi-Agent System](../project-c-multi-agent/) - Agent collaboration
- [Project D: Computer Vision](../project-d-computer-vision/) - Image processing

**Advanced Projects**:
- [Project 07: Agentic AI Platform](../../advanced-data-ai-projects/project-07-agentic-ai/) - Production agents
- [Project 08: LLM Fine-tuning Platform](../../advanced-data-ai-projects/project-08-llm-finetuning/) - Training infrastructure

**Next Steps**:
1. Apply RL to custom problems
2. Implement advanced algorithms (Rainbow, IMPALA)
3. Build multi-agent systems
4. Deploy RL agents to production
5. Contribute to open-source RL projects

---

**Ready to train your first RL agent?** Start with the DQN CartPole example and work through the implementation plan. Good luck! 🚀
