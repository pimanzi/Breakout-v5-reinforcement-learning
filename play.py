"""
===============================================================================
                    BREAKOUT-V5 DQN PLAYING SCRIPT
===============================================================================
Task 2: Playing Script (play.py)

Objective:
- Load the trained DQN model
- Play Breakout-v5 with the trained agent
- Use greedy policy (deterministic actions)
- Display the game in real-time
===============================================================================
"""

import gymnasium as gym
import argparse
import os
import numpy as np
from datetime import datetime

# Import ale_py to register ALE environments with Gymnasium
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import RecordVideo


# ===============================================================================
# ENVIRONMENT SETUP
# ===============================================================================

def make_atari_env(env_id="ALE/Breakout-v5", n_stack=4, render_mode="human", record_video=False, video_folder="./videos"):
    """
    Create Atari environment for playing.
    
    Args:
        env_id: Gymnasium environment ID
        n_stack: Number of frames to stack
        render_mode: Rendering mode (human for visualization, rgb_array for recording)
        record_video: Whether to record video
        video_folder: Folder to save videos
    
    Returns:
        Vectorized environment
    """
    def make_env():
        env = gym.make(env_id, render_mode="rgb_array" if record_video else render_mode)
        
        # Add video recording wrapper if requested
        if record_video:
            os.makedirs(video_folder, exist_ok=True)
            env = RecordVideo(
                env, 
                video_folder=video_folder,
                episode_trigger=lambda x: True,  # Record all episodes
                name_prefix="breakout-gameplay"
            )
        
        env = AtariWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=n_stack)
    return env


# ===============================================================================
# PLAYING FUNCTION
# ===============================================================================

def play_breakout(model_path, num_episodes=5, record_video=False, video_folder="./videos"):
    """
    Play Breakout with trained DQN agent.
    
    Args:
        model_path: Path to saved model (.zip file)
        num_episodes: Number of episodes to play
        record_video: Whether to record gameplay as video
        video_folder: Folder to save recorded videos
    """
    print("="*80)
    print("          LOADING TRAINED MODEL")
    print("="*80)
    print(f"Model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"\nError: Model file not found at {model_path}")
        print("\nAvailable models:")
        models_dir = "./models"
        if os.path.exists(models_dir):
            for exp in os.listdir(models_dir):
                exp_path = os.path.join(models_dir, exp)
                if os.path.isdir(exp_path):
                    print(f"  - {exp}/")
                    for file in os.listdir(exp_path):
                        if file.endswith(".zip"):
                            print(f"    - {file}")
        return
    
    # Load the trained model
    print("Loading model...")
    model = DQN.load(model_path)
    print("Model loaded successfully!")
    print("="*80)
    print()
    
    # Create environment with rendering
    if record_video:
        print(f"Creating environment with video recording...")
        print(f"Videos will be saved to: {video_folder}")
    else:
        print("Creating environment with rendering...")
    
    env = make_atari_env("ALE/Breakout-v5", n_stack=4, render_mode="human", 
                         record_video=record_video, video_folder=video_folder)
    print("Environment created!")
    print()
    
    # Play episodes
    print("="*80)
    print("          STARTING GAMEPLAY")
    print("="*80)
    print(f"Playing {num_episodes} episodes")
    if record_video:
        print(f"Recording video to: {video_folder}")
    print("Press Ctrl+C to stop at any time")
    print("="*80)
    print()
    
    episode_rewards = []
    episode_lengths = []
    
    try:
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print("-" * 40)
            
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Use deterministic policy (greedy - highest Q-value)
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1} finished")
            print(f"  Reward: {episode_reward}")
            print(f"  Length: {episode_length} steps")
        
        # Print summary
        print("\n" + "="*80)
        print("          SUMMARY")
        print("="*80)
        print(f"Episodes played: {num_episodes}")
        print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
        print(f"Best reward: {max(episode_rewards)}")
        print(f"Worst reward: {min(episode_rewards)}")
        print(f"Average length: {sum(episode_lengths) / len(episode_lengths):.2f} steps")
        
        if record_video:
            print(f"\n📹 Videos saved to: {video_folder}")
            print("Video files:")
            if os.path.exists(video_folder):
                for file in sorted(os.listdir(video_folder)):
                    if file.endswith(".mp4"):
                        file_path = os.path.join(video_folder, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                        print(f"  - {file} ({file_size:.2f} MB)")
        
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted by user")
    
    finally:
        env.close()
        print("\nEnvironment closed")


# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def main():
    """Main function to play with trained agent."""
    
    parser = argparse.ArgumentParser(
        description="Play Breakout-v5 with trained DQN agent and optionally record video"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/experiment_1/best_model.zip",
        help="Path to trained model (default: ./models/experiment_1/best_model.zip)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5)"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record gameplay as video files"
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="./videos",
        help="Folder to save recorded videos (default: ./videos)"
    )
    
    args = parser.parse_args()
    
    play_breakout(args.model, args.episodes, args.record, args.video_folder)


# ===============================================================================
# SCRIPT ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    main()
