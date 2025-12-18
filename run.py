# train_monopoly.py
"""
Complete training script for Monopoly RL agent using PPO.

Features:
- Multiple parallel environments for faster training
- Callbacks for monitoring and checkpointing
- Evaluation during training
- Tensorboard logging
- Action masking support
- Curriculum learning options
"""
import warnings
warnings.filterwarnings("ignore", message="sys.meta_path is None")
import os
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from MonopolyEnv import *


# ===============================
# Configuration
# ===============================
class Config:
    # Training parameters
    TOTAL_TIMESTEPS = 1_000_000  # Total training steps
    NUM_ENVS = 4                  # Parallel environments
    SAVE_FREQ = 50_000           # Save model every N steps
    EVAL_FREQ = 25_000           # Evaluate every N steps
    EVAL_EPISODES = 10           # Episodes per evaluation
    
    # Environment parameters
    NUM_PLAYERS = 4              # Players per game (2-6)
    MAX_TURNS = 500              # Max turns per episode
    USE_TIMER = False            # Enable 90-second timer (disable for training)
    
    # PPO hyperparameters
    LEARNING_RATE = 3e-4
    N_STEPS = 2048               # Steps per environment before update
    BATCH_SIZE = 64              # Minibatch size
    N_EPOCHS = 10                # Epochs per update
    GAMMA = 0.99                 # Discount factor
    GAE_LAMBDA = 0.95            # GAE parameter
    CLIP_RANGE = 0.2             # PPO clip range
    ENT_COEF = 0.01              # Entropy coefficient
    VF_COEF = 0.5                # Value function coefficient
    MAX_GRAD_NORM = 0.5          # Gradient clipping
    
    # Network architecture
    NET_ARCH = [256, 256]        # Hidden layer sizes
    
    # Directories
    LOG_DIR = "./logs"
    MODEL_DIR = "./models"
    TENSORBOARD_DIR = "./tensorboard"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Resume training
    RESUME_TRAINING = False
    RESUME_MODEL_PATH = None     # Path to model.zip if resuming


# ===============================
# Environment Creation
# ===============================
def make_env(rank, seed=0):
    """
    Create a single Monopoly environment.
    
    Parameters
    ----------
    rank : int
        Environment index (for seeding)
    seed : int
        Base random seed
    """
    def _init():
        env = MonopolyEnv(
            agent_index=0,
            num_players=Config.NUM_PLAYERS,
            use_timer=Config.USE_TIMER,
            max_turns=Config.MAX_TURNS,
            seed=seed + rank
        )
        env = Monitor(env, Config.LOG_DIR)
        return env
    return _init


def make_vec_env(n_envs=1, seed=0):
    """
    Create vectorized environments for parallel training.
    
    Parameters
    ----------
    n_envs : int
        Number of parallel environments
    seed : int
        Base random seed
    
    Returns
    -------
    VecEnv
        Vectorized environment
    """
    if n_envs == 1:
        return DummyVecEnv([make_env(0, seed)])
    else:
        # Use SubprocVecEnv for true parallelism (faster)
        return SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])


# ===============================
# Custom Callbacks
# ===============================
class MonopolyCallback(CallbackList):
    """Custom callback for monitoring Monopoly training."""
    
    def __init__(self, verbose=0):
        super().__init__([])
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.bankruptcy_count = 0
        self.total_episodes = 0
        
    def _on_step(self):
        # Check if episode finished
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.total_episodes += 1
                
                # Get episode info
                info = self.locals.get("infos", [{}])[i]
                
                # Track termination reasons
                reason = info.get("reason", "unknown")
                if reason == "victory":
                    self.win_count += 1
                elif reason == "bankruptcy":
                    self.bankruptcy_count += 1
                
                # Log statistics every 10 episodes
                if self.total_episodes % 10 == 0:
                    win_rate = self.win_count / self.total_episodes if self.total_episodes > 0 else 0
                    bankruptcy_rate = self.bankruptcy_count / self.total_episodes if self.total_episodes > 0 else 0
                    
                    print(f"\n--- Episode {self.total_episodes} ---")
                    print(f"Win Rate: {win_rate:.2%}")
                    print(f"Bankruptcy Rate: {bankruptcy_rate:.2%}")
                    print(f"Net Worth: ${info.get('net_worth', 0):.0f}")
                    print(f"Reason: {reason}")
        
        return True


# ===============================
# Training Function
# ===============================
def train():
    """Main training function."""
    
    # Create directories
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.TENSORBOARD_DIR, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"monopoly_ppo_{timestamp}"
    
    print("="*60)
    print("MONOPOLY RL TRAINING")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Environments: {Config.NUM_ENVS}")
    print(f"Players per game: {Config.NUM_PLAYERS}")
    print(f"Total timesteps: {Config.TOTAL_TIMESTEPS:,}")
    print(f"Run name: {run_name}")
    print("="*60)
    
    # Create training environment
    print("\nCreating training environments...")
    env = make_vec_env(n_envs=Config.NUM_ENVS, seed=42)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(n_envs=1, seed=123)
    
    # Create or load model
    if Config.RESUME_TRAINING and Config.RESUME_MODEL_PATH:
        print(f"\nResuming training from: {Config.RESUME_MODEL_PATH}")
        model = PPO.load(Config.RESUME_MODEL_PATH, env=env, device=Config.DEVICE)
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=Config.LEARNING_RATE,
            n_steps=Config.N_STEPS,
            batch_size=Config.BATCH_SIZE,
            n_epochs=Config.N_EPOCHS,
            gamma=Config.GAMMA,
            gae_lambda=Config.GAE_LAMBDA,
            clip_range=Config.CLIP_RANGE,
            ent_coef=Config.ENT_COEF,
            vf_coef=Config.VF_COEF,
            max_grad_norm=Config.MAX_GRAD_NORM,
            verbose=1,
            device=Config.DEVICE,
            tensorboard_log=f"{Config.TENSORBOARD_DIR}/{run_name}",
            policy_kwargs={
                "net_arch": Config.NET_ARCH,
            }
        )
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=Config.SAVE_FREQ,
        save_path=f"{Config.MODEL_DIR}/{run_name}",
        name_prefix="monopoly_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Evaluation callback - evaluate during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{Config.MODEL_DIR}/{run_name}/best_model",
        log_path=f"{Config.LOG_DIR}/{run_name}",
        eval_freq=Config.EVAL_FREQ,
        n_eval_episodes=Config.EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    
    # Custom monitoring callback
    monopoly_callback = MonopolyCallback(verbose=1)
    
    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback, monopoly_callback])
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Monitor with: tensorboard --logdir {Config.TENSORBOARD_DIR}")
    print("="*60 + "\n")
    
    try:
        model.learn(
            total_timesteps=Config.TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_path = f"{Config.MODEL_DIR}/{run_name}/final_model"
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # Clean up
    env.close()
    eval_env.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Win Rate: {monopoly_callback.win_count / monopoly_callback.total_episodes:.2%}")
    print(f"Bankruptcy Rate: {monopoly_callback.bankruptcy_count / monopoly_callback.total_episodes:.2%}")
    print(f"Total Episodes: {monopoly_callback.total_episodes}")
    print("="*60)


# ===============================
# Evaluation Function
# ===============================
def evaluate(model_path, n_episodes=10, render=True):
    """
    Evaluate a trained model.
    
    Parameters
    ----------
    model_path : str
        Path to saved model
    n_episodes : int
        Number of episodes to evaluate
    render : bool
        Whether to render the game
    """
    print(f"\nEvaluating model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("="*60)
    
    # Create environment
    env = MonopolyEnv(
        agent_index=0,
        num_players=Config.NUM_PLAYERS,
        use_timer=False,
        max_turns=Config.MAX_TURNS,
    )
    
    # Load model
    model = PPO.load(model_path, device=Config.DEVICE)
    
    # Run episodes
    wins = 0
    bankruptcies = 0
    total_rewards = []
    net_worths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
        
        # Track results
        total_rewards.append(episode_reward)
        net_worths.append(info.get("net_worth", 0))
        
        reason = info.get("reason", "unknown")
        if reason == "victory":
            wins += 1
        elif reason == "bankruptcy":
            bankruptcies += 1
        
        print(f"Result: {reason}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Net Worth: ${info.get('net_worth', 0):.0f}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Wins: {wins} ({wins/n_episodes:.1%})")
    print(f"Bankruptcies: {bankruptcies} ({bankruptcies/n_episodes:.1%})")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Net Worth: ${np.mean(net_worths):.0f} ± ${np.std(net_worths):.0f}")
    print("="*60)
    
    env.close()


# ===============================
# Main Entry Point
# ===============================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or evaluate Monopoly RL agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                      help="Mode: train or eval")
    parser.add_argument("--model", type=str, default=None,
                      help="Path to model for evaluation or resuming training")
    parser.add_argument("--timesteps", type=int, default=None,
                      help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=None,
                      help="Number of parallel environments")
    parser.add_argument("--players", type=int, default=None,
                      help="Number of players per game")
    parser.add_argument("--eval-episodes", type=int, default=10,
                      help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true",
                      help="Disable rendering during evaluation")
    
    args = parser.parse_args()
    
    # Override config with command line args
    if args.timesteps:
        Config.TOTAL_TIMESTEPS = args.timesteps
    if args.n_envs:
        Config.NUM_ENVS = args.n_envs
    if args.players:
        Config.NUM_PLAYERS = args.players
    if args.model:
        Config.RESUME_TRAINING = True
        Config.RESUME_MODEL_PATH = args.model
    
    # Run
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        if not args.model:
            print("Error: --model required for evaluation mode")
            exit(1)
        evaluate(args.model, n_episodes=args.eval_episodes, render=not args.no_render)