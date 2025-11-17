"""
===============================================================================
                    BREAKOUT-V5 DQN TRAINING SCRIPT
===============================================================================
Task 1: Training Script (train.py)

Objective:
- Train a DQN agent on Atari Breakout-v5 environment
- Experiment with different hyperparameters
- Save the best performing model
- Log training metrics

Technologies:
- Stable Baselines3 (DQN implementation)
- Gymnasium (Atari environments)
- TensorBoard (for logging)

HOW TO USE:
-----------
1. Change EXPERIMENT_TO_TRAIN variable below (lines 35-36)
2. Run: python train.py
3. Wait for training to complete (6-10 hours for CNN, 8-14 hours for MLP)
4. Check results with TensorBoard: tensorboard --logdir ./logs
5. Test model: python play.py --model ./models/experiment_X/best_model.zip
6. Train next experiment: Change EXPERIMENT_TO_TRAIN and run again

QUICK START:
-----------
# Train baseline CNN (recommended first)
EXPERIMENT_TO_TRAIN = "experiment_1"
python train.py

# Train MLP for comparison
EXPERIMENT_TO_TRAIN = "experiment_10"
python train.py

# Quick test (100k timesteps, ~1 hour)
TOTAL_TIMESTEPS = 100000
python train.py

Author: Generated for Breakout-v5 RL Project
===============================================================================
"""

import gymnasium as gym
import numpy as np
import os
import argparse
from datetime import datetime

# Stable Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# ===============================================================================
# HYPERPARAMETER CONFIGURATIONS (10 Experiments)
# ===============================================================================

EXPERIMENTS = {
    "experiment_1": {
        "name": "Baseline",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "learning_starts": 10000,
        "train_freq": 4
    },
    "experiment_2": {
        "name": "Higher Learning Rate",
        "learning_rate": 5e-4,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "learning_starts": 10000,
        "train_freq": 4
    },
    "experiment_3": {
        "name": "Lower Learning Rate",
        "learning_rate": 5e-5,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "learning_starts": 10000,
        "train_freq": 4
    },
    "experiment_4": {
        "name": "Larger Buffer",
        "learning_rate": 1e-4,
        "buffer_size": 500000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "learning_starts": 10000,
        "train_freq": 4
    },
    "experiment_5": {
        "name": "Larger Batch Size",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "learning_starts": 10000,
        "train_freq": 4
    },
    "experiment_6": {
        "name": "Higher Gamma",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.995,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "learning_starts": 10000,
        "train_freq": 4
    },
    "experiment_7": {
        "name": "More Exploration",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "learning_starts": 10000,
        "train_freq": 4
    },
    "experiment_8": {
        "name": "Frequent Target Update",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 500,
        "learning_starts": 10000,
        "train_freq": 4
    },
    "experiment_9": {
        "name": "Early Training Start",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "learning_starts": 5000,
        "train_freq": 4
    },
    "experiment_10": {
        "name": "MLP Architecture (Comparison)",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "learning_starts": 10000,
        "train_freq": 4,
        "policy": "MlpPolicy"  # Using MLP instead of CNN
    }
}


# ===============================================================================
# ENVIRONMENT SETUP
# ===============================================================================

def make_atari_env(env_id="ALE/Breakout-v5", n_stack=4):
    """
    Create Atari environment with standard preprocessing.
    
    Args:
        env_id: Gymnasium environment ID
        n_stack: Number of frames to stack
    
    Returns:
        Vectorized environment
    """
    def make_env():
        env = gym.make(env_id)
        env = AtariWrapper(env)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=n_stack)
    return env


# ===============================================================================
# TRAINING FUNCTION
# ===============================================================================

def train_dqn(experiment_name, hyperparams, total_timesteps=1000000, 
              save_dir="./models", log_dir="./logs"):
    """
    Train a DQN agent with specified hyperparameters.
    
    Args:
        experiment_name: Name of the experiment
        hyperparams: Dictionary of hyperparameters
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        log_dir: Directory to save logs
    
    Returns:
        Trained model
    """
    print("="*80)
    print(f"Starting Training: {experiment_name}")
    print(f"Description: {hyperparams['name']}")
    print("="*80)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create experiment-specific paths
    exp_save_dir = os.path.join(save_dir, experiment_name)
    exp_log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)
    
    # Print hyperparameters
    print("\nHyperparameters:")
    for key, value in hyperparams.items():
        if key != "name":
            print(f"  {key}: {value}")
    print()
    
    # Create environment
    print("Creating environment...")
    env = make_atari_env("ALE/Breakout-v5", n_stack=4)
    
    # Create evaluation environment
    eval_env = make_atari_env("ALE/Breakout-v5", n_stack=4)
    
    # Create DQN model
    # Use MlpPolicy for experiment_10, CnnPolicy for all others
    policy_type = hyperparams.get("policy", "CnnPolicy")
    print(f"Creating DQN model with {policy_type}...")
    
    model = DQN(
        policy=policy_type,
        env=env,
        learning_rate=hyperparams["learning_rate"],
        buffer_size=hyperparams["buffer_size"],
        batch_size=hyperparams["batch_size"],
        gamma=hyperparams["gamma"],
        exploration_fraction=hyperparams["exploration_fraction"],
        exploration_final_eps=hyperparams["exploration_final_eps"],
        target_update_interval=hyperparams["target_update_interval"],
        learning_starts=hyperparams["learning_starts"],
        train_freq=hyperparams["train_freq"],
        tensorboard_log=exp_log_dir,
        verbose=1
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=exp_save_dir,
        log_path=exp_log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=exp_save_dir,
        name_prefix="checkpoint"
    )
    
    # Train the model
    print(f"\nTraining for {total_timesteps} timesteps...")
    print("This may take several hours depending on your hardware.")
    print("Training progress will be displayed below:\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
        tb_log_name=experiment_name
    )
    
    # Save final model
    final_model_path = os.path.join(exp_save_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Clean up
    env.close()
    eval_env.close()
    
    print(f"\nTraining completed: {experiment_name}")
    print("="*80)
    print()
    
    return model


# ===============================================================================
# CONFIGURATION - CHANGE THESE TO TRAIN DIFFERENT EXPERIMENTS
# ===============================================================================

# Which experiment to train (change this to train different experiments)
EXPERIMENT_TO_TRAIN = "experiment_1"  # Options: experiment_1 to experiment_10

# Training configuration
TOTAL_TIMESTEPS = 500000  # 1M timesteps (recommended), use 100000 for quick testing
SAVE_DIR = "./models"
LOG_DIR = "./logs"


# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """Main function to run training experiments."""
    
    # Use command-line arguments if provided, otherwise use configuration above
    parser = argparse.ArgumentParser(
        description="Train DQN agent on Breakout-v5"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=EXPERIMENT_TO_TRAIN,
        choices=list(EXPERIMENTS.keys()),
        help=f"Which experiment to run (default: {EXPERIMENT_TO_TRAIN})"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=SAVE_DIR,
        help=f"Directory to save models (default: {SAVE_DIR})"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=LOG_DIR,
        help=f"Directory to save logs (default: {LOG_DIR})"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("          BREAKOUT-V5 DQN TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment: {args.experiment}")
    print(f"Description: {EXPERIMENTS[args.experiment]['name']}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Models will be saved to: {args.save_dir}/{args.experiment}/")
    print(f"Best model will be: {args.save_dir}/{args.experiment}/best_model.zip")
    print(f"Logs will be saved to: {args.log_dir}/{args.experiment}/")
    print("="*80)
    print()
    
    # Run single experiment
    hyperparams = EXPERIMENTS[args.experiment]
    model = train_dqn(
        experiment_name=args.experiment,
        hyperparams=hyperparams,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    print("\n" + "="*80)
    print("          TRAINING COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTrained experiment: {args.experiment}")
    print(f"Best model saved at: {args.save_dir}/{args.experiment}/best_model.zip")
    print(f"\nNext steps:")
    print(f"1. View training progress:")
    print(f"   tensorboard --logdir {args.log_dir}")
    print(f"\n2. Test the trained model:")
    print(f"   python play.py --model {args.save_dir}/{args.experiment}/best_model.zip")
    print(f"\n3. To train another experiment:")
    print(f"   - Change EXPERIMENT_TO_TRAIN in train.py")
    print(f"   - Or run: python train.py --experiment experiment_2")
    print("="*80)


# ===============================================================================
# SCRIPT ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    main()
