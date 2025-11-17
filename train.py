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
1. Change EXPERIMENT_TO_TRAIN variable below (line 75)
2. Run: python train.py
3. Wait for training to complete (progress shown with ETA)
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

# Generate comparison report after training multiple experiments
python train.py --compare

Author: Generated for Breakout-v5 RL Project
===============================================================================
"""

import gymnasium as gym
import numpy as np
import os
import argparse
import json
from datetime import datetime, timedelta
import time

# Import ale_py to register ALE environments with Gymnasium
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

# Stable Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


# ===============================================================================
# HYPERPARAMETER CONFIGURATIONS (10 Experiments)
# ===============================================================================

EXPERIMENTS = {
    "experiment_1": {
        "name": "Baseline CNN",
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
        "name": "Higher Learning Rate (CNN)",
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
        "name": "Lower Learning Rate (CNN)",
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
        "name": "Larger Buffer (CNN)",
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
        "name": "Larger Batch Size (CNN)",
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
        "name": "Higher Gamma (CNN)",
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
        "name": "More Exploration (CNN)",
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
        "name": "Frequent Target Update (CNN)",
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
        "name": "Early Training Start (CNN)",
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
# PROGRESS TRACKING CALLBACK
# ===============================================================================

class DetailedProgressCallback(BaseCallback):
    """
    Enhanced callback that shows detailed progress with ETA.
    Shows: progress bar, percentage, speed, elapsed time, and ETA.
    """
    def __init__(self, total_timesteps, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.start_time = None
        self.last_time = None
        self.last_step = 0
        
    def _on_training_start(self):
        """Called when training starts"""
        self.start_time = time.time()
        self.last_time = self.start_time
        print("\n" + "="*80)
        print("TRAINING PROGRESS TRACKER")
        print("="*80)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Progress will be updated every {self.check_freq:,} steps")
        print("="*80 + "\n")
        
    def _on_step(self):
        """Called at each step during training"""
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            current_time = time.time()
            
            # Calculate progress
            progress = (self.num_timesteps / self.total_timesteps) * 100
            
            # Calculate elapsed time
            elapsed_seconds = current_time - self.start_time
            elapsed = str(timedelta(seconds=int(elapsed_seconds)))
            
            # Calculate speed
            time_diff = current_time - self.last_time
            steps_diff = self.num_timesteps - self.last_step
            speed = steps_diff / time_diff if time_diff > 0 else 0
            
            # Calculate ETA
            if speed > 0:
                remaining_steps = self.total_timesteps - self.num_timesteps
                eta_seconds = remaining_steps / speed
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "Calculating..."
            
            # Print progress bar
            bar_length = 40
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print(f"\n{'='*80}")
            print(f"Step: {self.num_timesteps:,} / {self.total_timesteps:,}")
            print(f"[{bar}] {progress:.1f}%")
            print(f"Speed: {speed:.0f} steps/sec")
            print(f"Elapsed: {elapsed}")
            print(f"ETA: {eta}")
            print(f"{'='*80}")
            
            # Update tracking variables
            self.last_time = current_time
            self.last_step = self.num_timesteps
            
        return True


# ===============================================================================
# RESULTS SAVING FUNCTIONS
# ===============================================================================

def save_training_results(experiment_name, model, hyperparams, 
                         total_timesteps, save_dir, log_dir, 
                         start_time, end_time):
    """
    Save comprehensive training results including:
    - Hyperparameters used
    - Training duration
    - Best model performance
    - Final evaluation results
    """
    exp_save_dir = os.path.join(save_dir, experiment_name)
    exp_log_dir = os.path.join(log_dir, experiment_name)
    
    # Calculate training duration
    duration = end_time - start_time
    duration_hours = duration.total_seconds() / 3600
    
    # Load evaluation results
    eval_results_path = os.path.join(exp_log_dir, "evaluations.npz")
    eval_data = None
    best_mean_reward = None
    
    if os.path.exists(eval_results_path):
        eval_data = np.load(eval_results_path)
        best_mean_reward = float(np.max(eval_data['results']))
    
    # Create results dictionary
    results = {
        "experiment_name": experiment_name,
        "experiment_description": hyperparams["name"],
        "policy_type": hyperparams.get("policy", "CnnPolicy"),
        "training_info": {
            "total_timesteps": total_timesteps,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_hours": round(duration_hours, 2),
            "duration_formatted": str(duration)
        },
        "hyperparameters": {
            k: v for k, v in hyperparams.items() 
            if k not in ["name", "policy"]
        },
        "performance": {
            "best_mean_reward": best_mean_reward,
            "best_model_path": os.path.join(exp_save_dir, "best_model.zip"),
            "final_model_path": os.path.join(exp_save_dir, "final_model.zip")
        },
        "file_locations": {
            "models_directory": exp_save_dir,
            "logs_directory": exp_log_dir,
            "tensorboard_logs": exp_log_dir,
            "evaluation_results": eval_results_path
        }
    }
    
    # Save results as JSON
    results_file = os.path.join(exp_save_dir, "training_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save results as human-readable text
    results_txt_file = os.path.join(exp_save_dir, "training_results.txt")
    with open(results_txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"TRAINING RESULTS: {experiment_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Description: {hyperparams['name']}\n")
        f.write(f"Policy Type: {hyperparams.get('policy', 'CnnPolicy')}\n\n")
        
        f.write("TRAINING INFORMATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Timesteps: {total_timesteps:,}\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration} ({duration_hours:.2f} hours)\n\n")
        
        f.write("HYPERPARAMETERS:\n")
        f.write("-" * 80 + "\n")
        for key, value in hyperparams.items():
            if key not in ["name", "policy"]:
                f.write(f"{key:25s}: {value}\n")
        f.write("\n")
        
        f.write("PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        if best_mean_reward is not None:
            f.write(f"Best Mean Reward: {best_mean_reward:.2f}\n")
        f.write(f"Best Model: {os.path.join(exp_save_dir, 'best_model.zip')}\n")
        f.write(f"Final Model: {os.path.join(exp_save_dir, 'final_model.zip')}\n\n")
        
        f.write("FILE LOCATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Models Directory: {exp_save_dir}\n")
        f.write(f"Logs Directory: {exp_log_dir}\n")
        f.write(f"TensorBoard Logs: {exp_log_dir}\n")
        f.write(f"Evaluation Results: {eval_results_path}\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. View training progress:\n")
        f.write(f"   tensorboard --logdir {log_dir}\n\n")
        f.write(f"2. Test the trained model:\n")
        f.write(f"   python play.py --model {os.path.join(exp_save_dir, 'best_model.zip')}\n\n")
        f.write(f"3. Load evaluation data in Python:\n")
        f.write(f"   import numpy as np\n")
        f.write(f"   data = np.load('{eval_results_path}')\n")
        f.write(f"   rewards = data['results']\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"Human-readable results saved to: {results_txt_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    if best_mean_reward is not None:
        print(f"Best Mean Reward: {best_mean_reward:.2f}")
    print(f"Training Duration: {duration_hours:.2f} hours")
    print(f"Models saved in: {exp_save_dir}")
    print(f"Logs saved in: {exp_log_dir}")
    print("="*80)
    
    return results


def create_comparison_report(save_dir="./models"):
    """
    Create a comparison report of all trained experiments.
    """
    print("\nGenerating comparison report...")
    
    all_results = []
    
    # Load results from all experiments
    for exp_name in EXPERIMENTS.keys():
        results_file = os.path.join(save_dir, exp_name, "training_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                all_results.append(results)
    
    if not all_results:
        print("No training results found. Train at least one experiment first.")
        return
    
    # Sort by best mean reward (descending)
    all_results.sort(
        key=lambda x: x['performance']['best_mean_reward'] or 0, 
        reverse=True
    )
    
    # Save comparison report
    comparison_file = os.path.join(save_dir, "experiments_comparison.txt")
    with open(comparison_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENTS COMPARISON REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments: {len(all_results)}\n\n")
        
        # Summary table
        f.write("PERFORMANCE RANKING:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Experiment':<15} {'Description':<25} {'Best Reward':<15} {'Duration (h)':<12}\n")
        f.write("-" * 80 + "\n")
        
        for rank, result in enumerate(all_results, 1):
            exp_name = result['experiment_name']
            desc = result['experiment_description'][:23]
            reward = result['performance']['best_mean_reward']
            reward_str = f"{reward:.2f}" if reward is not None else "N/A"
            duration = result['training_info']['duration_hours']
            
            f.write(f"{rank:<6} {exp_name:<15} {desc:<25} {reward_str:<15} {duration:<12.2f}\n")
        
        f.write("\n\n")
        
        # Detailed results for each experiment
        f.write("DETAILED RESULTS:\n")
        f.write("="*80 + "\n\n")
        
        for result in all_results:
            f.write(f"Experiment: {result['experiment_name']}\n")
            f.write(f"Description: {result['experiment_description']}\n")
            f.write(f"Policy: {result['policy_type']}\n")
            
            if result['performance']['best_mean_reward'] is not None:
                f.write(f"Best Mean Reward: {result['performance']['best_mean_reward']:.2f}\n")
            
            f.write(f"Training Duration: {result['training_info']['duration_hours']:.2f} hours\n")
            f.write(f"Total Timesteps: {result['training_info']['total_timesteps']:,}\n")
            
            f.write("\nHyperparameters:\n")
            for key, value in result['hyperparameters'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "-"*80 + "\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"Comparison report saved to: {comparison_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("EXPERIMENTS COMPARISON")
    print("="*80)
    print(f"{'Rank':<6} {'Experiment':<15} {'Description':<25} {'Best Reward':<15}")
    print("-" * 80)
    
    for rank, result in enumerate(all_results, 1):
        exp_name = result['experiment_name']
        desc = result['experiment_description'][:23]
        reward = result['performance']['best_mean_reward']
        reward_str = f"{reward:.2f}" if reward is not None else "N/A"
        
        print(f"{rank:<6} {exp_name:<15} {desc:<25} {reward_str:<15}")
    
    print("="*80)


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
    Includes progress tracking with ETA and comprehensive results saving.
    """
    training_start_time = datetime.now()
    
    print("="*80)
    print(f"Starting Training: {experiment_name}")
    print(f"Description: {hyperparams['name']}")
    print(f"Start Time: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
        if key not in ["name", "policy"]:
            print(f"  {key}: {value}")
    print()
    
    # Create environment
    print("Creating environment...")
    env = make_atari_env("ALE/Breakout-v5", n_stack=4)
    eval_env = make_atari_env("ALE/Breakout-v5", n_stack=4)
    
    # Create DQN model
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
    
    # Create callbacks with progress tracking
    print("\nSetting up callbacks...")
    
    # Progress tracker with ETA
    progress_callback = DetailedProgressCallback(
        total_timesteps=total_timesteps,
        check_freq=5000,  # Update every 5000 steps
        verbose=1
    )
    
    # Evaluation callback - saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=exp_save_dir,
        log_path=exp_log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=exp_save_dir,
        name_prefix="checkpoint",
        verbose=1
    )
    
    # Train the model
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("Progress updates will appear every 5,000 steps")
    print("Evaluations will run every 10,000 steps")
    print("Checkpoints will save every 50,000 steps")
    print("\nTraining in progress...\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, eval_callback, checkpoint_callback],
        log_interval=100,
        tb_log_name=experiment_name
    )
    
    # Training completed
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    
    # Save final model
    final_model_path = os.path.join(exp_save_dir, "final_model.zip")
    model.save(final_model_path)
    
    # Clean up
    env.close()
    eval_env.close()
    
    # Save comprehensive results
    print("\nSaving training results...")
    save_training_results(
        experiment_name=experiment_name,
        model=model,
        hyperparams=hyperparams,
        total_timesteps=total_timesteps,
        save_dir=save_dir,
        log_dir=log_dir,
        start_time=training_start_time,
        end_time=training_end_time
    )
    
    # Print completion summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Experiment: {experiment_name}")
    print(f"Start Time: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {training_duration}")
    print(f"Duration (hours): {training_duration.total_seconds() / 3600:.2f}")
    print(f"\nModels saved:")
    print(f"  Best model: {exp_save_dir}/best_model.zip")
    print(f"  Final model: {final_model_path}")
    print("="*80)
    print()
    
    return model


# ===============================================================================
# CONFIGURATION - CHANGE THESE TO TRAIN DIFFERENT EXPERIMENTS
# ===============================================================================

# Which experiment to train (change this to train different experiments)
EXPERIMENT_TO_TRAIN = "experiment_1"  # Options: experiment_1 to experiment_10

# Training configuration
TOTAL_TIMESTEPS = 100000  # 100k timesteps for quick testing
SAVE_DIR = "./models"
LOG_DIR = "./logs"


# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """Main function to run training experiments."""
    
    # Use command-line arguments if provided, otherwise use configuration above
    parser = argparse.ArgumentParser(
        description="Train DQN agent on Breakout-v5 with progress tracking and results saving"
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison report of all trained experiments and exit"
    )
    
    args = parser.parse_args()
    
    # If comparison flag is set, generate report and exit
    if args.compare:
        create_comparison_report(save_dir=args.save_dir)
        return
    
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
    print(f"Results saved at: {args.save_dir}/{args.experiment}/training_results.txt")
    print(f"\nNext steps:")
    print(f"1. View training progress:")
    print(f"   tensorboard --logdir {args.log_dir}")
    print(f"\n2. Test the trained model:")
    print(f"   python play.py --model {args.save_dir}/{args.experiment}/best_model.zip")
    print(f"\n3. To train another experiment:")
    print(f"   - Change EXPERIMENT_TO_TRAIN in train.py (line 457)")
    print(f"   - Or run: python train.py --experiment experiment_2")
    print(f"\n4. Generate comparison report (after training multiple experiments):")
    print(f"   python train.py --compare")
    print("="*80)


# ===============================================================================
# SCRIPT ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    main()