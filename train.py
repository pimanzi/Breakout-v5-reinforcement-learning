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
        "name": "More Frequent Training",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "learning_starts": 10000,
        "train_freq": 8
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
    print("Creating DQN model...")
    model = DQN(
        policy="CnnPolicy",
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
# MAIN EXECUTION
# ===============================================================================

def main():
    """Main function to run training experiments."""
    
    parser = argparse.ArgumentParser(
        description="Train DQN agent on Breakout-v5"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment_1",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Which experiment to run (default: experiment_1, use 'all' for all experiments)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Total training timesteps (default: 1000000)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save logs (default: ./logs)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("          BREAKOUT-V5 DQN TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total timesteps per experiment: {args.timesteps}")
    print(f"Models will be saved to: {args.save_dir}")
    print(f"Logs will be saved to: {args.log_dir}")
    print("="*80)
    print()
    
    # Run experiments
    if args.experiment == "all":
        print("Running all 10 experiments...")
        print("This will take a very long time. Consider running experiments individually.")
        print()
        
        for exp_name, hyperparams in EXPERIMENTS.items():
            train_dqn(
                experiment_name=exp_name,
                hyperparams=hyperparams,
                total_timesteps=args.timesteps,
                save_dir=args.save_dir,
                log_dir=args.log_dir
            )
    else:
        # Run single experiment
        hyperparams = EXPERIMENTS[args.experiment]
        train_dqn(
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
    print(f"\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.log_dir}")
    print(f"\nTo use the trained model, run:")
    print(f"  python play.py")
    print("="*80)


# ===============================================================================
# SCRIPT ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    main()
