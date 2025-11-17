# Breakout-v5 Deep Q-Network (DQN) Training

## Introduction

This project implements a **Deep Q-Network (DQN)** agent to play the classic Atari game **Breakout** using reinforcement learning. The agent learns to control the paddle and break bricks by maximizing its cumulative reward through trial and error.

## 🎮 Demo Video

**▶️ [Watch the Trained Agent Playing Breakout](https://drive.google.com/file/d/1HQjT4W-yzhsZW2oywu1-nEt9H1hCKqJA/view?usp=sharing)**

See the trained DQN agent in action! This video demonstrates the agent's learned behavior after training with reinforcement learning.

---

## 📊 Results and Analysis

### Team Member: Placide Imanzi Kabisa

This section presents the experimental results from training 10 different DQN configurations on Atari Breakout-v5. Nine experiments used CNN architecture with varying hyperparameters, and one used MLP architecture for comparison.

---

### Hyperparameter Configuration and Results Table

| Experiment | Architecture   | Learning Rate | Gamma     | Batch Size | Epsilon Start | Epsilon End | Buffer Size | Best Reward |
| ---------- | -------------- | ------------- | --------- | ---------- | ------------- | ----------- | ----------- | ----------- |
| **Exp 1**  | CNN (Baseline) | 0.0001        | 0.99      | 32         | 1.0           | 0.01        | 100,000     | **4.0**     |
| **Exp 2**  | CNN            | **0.0005**    | 0.99      | 32         | 1.0           | 0.01        | 100,000     | **13.0** ⭐ |
| **Exp 3**  | CNN            | **0.00005**   | 0.99      | 32         | 1.0           | 0.01        | 100,000     | **6.0**     |
| **Exp 4**  | CNN            | 0.0001        | 0.99      | 32         | 1.0           | 0.01        | **500,000** | **4.0**     |
| **Exp 5**  | CNN            | 0.0001        | 0.99      | **64**     | 1.0           | 0.01        | 100,000     | **10.0**    |
| **Exp 6**  | CNN            | 0.0001        | **0.995** | 32         | 1.0           | 0.01        | 100,000     | **10.0**    |
| **Exp 7**  | CNN            | 0.0001        | 0.99      | 32         | 1.0           | **0.05**    | 100,000     | **12.0** 🥈 |
| **Exp 8**  | CNN            | 0.0001        | 0.99      | 32         | 1.0           | 0.01        | 100,000     | **6.0**     |
| **Exp 9**  | CNN            | 0.0001        | 0.99      | 32         | 1.0           | 0.01        | 100,000     | **5.0**     |
| **Exp 10** | **MLP**        | 0.0001        | 0.99      | 32         | 1.0           | 0.01        | 100,000     | **2.0** ❌  |

**Legend:**

- ⭐ = Best performing model
- 🥈 = Second best performing model
- ❌ = Worst performing model
- **Bold** values indicate the hyperparameter that was changed from baseline

---

### Key Findings

#### 1. **Best Performing Configuration**

**Experiment 2** achieved the highest reward of **13.0** using:

- **Higher Learning Rate**: 0.0005 (5x the baseline)
- CNN Architecture
- Standard exploration settings

**Why it worked:**
The higher learning rate allowed the agent to learn faster from experiences. With 500,000 training steps, this faster learning rate helped the agent discover better strategies more quickly without becoming unstable.

#### 2. **Second Best Configuration**

**Experiment 7** achieved **12.0** reward using:

- **Higher Final Epsilon**: 0.05 (5x the baseline)
- CNN Architecture
- More exploration throughout training

**Why it worked:**
Keeping epsilon at 0.05 (instead of 0.01) meant the agent continued exploring more even in later stages of training. This helped it discover better brick-breaking strategies instead of getting stuck in local optima.

#### 3. **Good Performers**

**Experiments 5 and 6** both achieved **10.0** reward by:

- **Exp 5**: Using larger batch size (64 vs 32) - More stable gradient updates
- **Exp 6**: Using higher gamma (0.995 vs 0.99) - Better long-term planning

These moderate improvements show that stable learning and forward-thinking help the agent play better.

#### 4. **Lower Performers**

**Experiments 1, 3, 4, 8, 9** achieved **4.0-6.0** rewards:

- **Exp 1 (Baseline)**: Standard settings, average performance
- **Exp 3**: Learning rate too low (0.00005) - learned too slowly
- **Exp 4**: Larger buffer didn't help with current training length
- **Exp 8 & 9**: Different update strategies showed mixed results

---

### CNN vs MLP Architecture Comparison

#### Performance Gap

| Architecture      | Best Reward | Performance |
| ----------------- | ----------- | ----------- |
| **CNN** (Best)    | 13.0        | Excellent   |
| **CNN** (Average) | 7.2         | Good        |
| **MLP**           | 2.0         | Poor ❌     |

**CNN is 6.5x better than MLP** on average!

#### Why CNN Outperforms MLP

1. **Spatial Understanding**: CNNs can see where the ball, paddle, and bricks are positioned. MLPs treat pixels as independent numbers and lose spatial information.

2. **Pattern Recognition**: CNNs learn features like "ball moving down" or "brick patterns". MLPs struggle to recognize these visual patterns.

3. **Efficient Learning**: CNNs share weights across the image, learning general patterns. MLPs need separate weights for each pixel, making learning much harder.

4. **Game-Specific Advantage**: Breakout requires understanding spatial relationships (ball trajectory, paddle position). CNNs naturally excel at this, while MLPs fail.

**Conclusion**: For visual game environments like Atari Breakout, CNN architecture is essential for good performance.

---

### Hyperparameter Analysis

#### Learning Rate Impact (Most Important)

- **Too Low (0.00005)**: Reward = 6.0 - Agent learns too slowly
- **Baseline (0.0001)**: Reward = 4.0 - Decent but not optimal
- **Higher (0.0005)**: Reward = 13.0 ⭐ - **BEST** - Fast learning without instability

**Lesson**: A higher learning rate (5x baseline) significantly improved performance because the agent could learn effective strategies faster within the training time available.

#### Exploration Strategy (Second Most Important)

- **Standard Epsilon (0.01)**: Average performance
- **Higher Epsilon (0.05)**: Reward = 12.0 🥈 - More exploration found better strategies

**Lesson**: Keeping some exploration (5% random actions) throughout training helped discover better ways to break bricks.

#### Batch Size and Gamma (Moderate Impact)

- **Larger Batch (64)**: Reward = 10.0 - Smoother learning
- **Higher Gamma (0.995)**: Reward = 10.0 - Better forward planning

**Lesson**: These improvements helped, but not as much as learning rate and exploration tuning.

#### Buffer Size (Minimal Impact)

- **5x Larger Buffer (500,000)**: Reward = 4.0 - No improvement

**Lesson**: With current training length, a larger replay buffer didn't help. The agent doesn't experience enough to fill the larger buffer effectively.

---

### Ranking and Recommendations

#### Performance Ranking

1. 🥇 **Experiment 2** (13.0) - Higher learning rate
2. 🥈 **Experiment 7** (12.0) - More exploration
3. 🥉 **Experiments 5 & 6** (10.0) - Larger batch / Higher gamma
4. **Experiments 3, 8** (6.0) - Lower learning rate / Different updates
5. **Experiments 1, 4, 9** (4.0-5.0) - Baseline and variations
6. ❌ **Experiment 10** (2.0) - MLP architecture

#### Best Practices for Breakout DQN

Based on these results, here are the recommended settings:

```python
# RECOMMENDED CONFIGURATION
learning_rate = 0.0005        # Higher LR for faster learning
buffer_size = 100000          # Standard size is sufficient
batch_size = 64               # Larger batch for stability
gamma = 0.995                 # Slightly higher for long-term planning
exploration_final_eps = 0.05  # Keep more exploration
architecture = "CnnPolicy"    # CNN is essential!
```

**Expected Performance**: 12-15 reward (based on Exp 2 and 7 results)

---

### Training Insights

#### Training Duration

- **Total Experiments**: 10
- **Total Training Time**: ~8.5 hours
- **Longest Experiment**: Experiment 2 (1.86 hours, 500k steps)
- **Shortest Experiment**: Experiment 10 (0.08 hours, 50k steps)

#### Learning Behavior Observed

1. **Early Training (0-50k steps)**: Agent learns basic paddle control
2. **Mid Training (50k-200k steps)**: Agent learns to hit the ball consistently
3. **Late Training (200k+ steps)**: Agent optimizes brick-breaking strategies

**Key Observation**: Experiments with 500k training steps (Exp 2, 3, 5, 6) generally performed better, showing that more training time helps when hyperparameters are well-tuned.

---

### Conclusions

1. **Learning Rate is Critical**: Increasing from 0.0001 to 0.0005 gave the biggest improvement (+225% reward increase from baseline).

2. **Exploration Matters**: Keeping epsilon at 0.05 instead of 0.01 helped the agent discover better strategies (+200% reward increase from baseline).

3. **CNN is Essential**: CNN achieved 6.5x better performance than MLP. For visual games, CNN architecture is non-negotiable.

4. **Training Time Helps**: Longer training (500k steps) consistently produced better results than shorter training (100k steps) when combined with good hyperparameters.

5. **Simple Changes, Big Impact**: Small hyperparameter adjustments (5x learning rate, 5x final epsilon) made huge performance differences, showing the importance of hyperparameter tuning.

**Final Recommendation**: Use Experiment 2's configuration (higher learning rate + CNN) as the foundation, and add Experiment 7's exploration strategy for best results.

---

## Table of Contents

- [Introduction](#introduction)
- [Demo Video](#-demo-video)
- [Results and Analysis](#-results-and-analysis)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Google Colab Setup](#google-colab-setup)
- [How to Use](#how-to-use)
  - [Training a Model](#training-a-model)
  - [Monitoring Training Progress](#monitoring-training-progress)
  - [Testing a Trained Model](#testing-a-trained-model)
- [File Locations](#file-locations)
- [Experiments](#experiments)
- [Troubleshooting](#troubleshooting)

---

### Key Features:

- 10 different hyperparameter experiments ( CNN + MLP )
- Automatic best model saving during training
- Real-time progress tracking with ETA
- TensorBoard integration for visualization
- Comprehensive results logging (JSON + human-readable format)
- Checkpoint system for crash recovery
- Comparison report generation across all experiments

### Environment:

- **Game:** Atari Breakout-v5
- **Objective:** Control paddle to bounce ball and break bricks
- **Action Space:** 4 discrete actions (NOOP, FIRE, RIGHT, LEFT)
- **Observation Space:** 84x84 grayscale image (4 frames stacked)
- **Max Score:** 864 points (clearing all bricks)

---

## Project Structure

```
Breakout-v5-reinforcement-learning/
├── train.py                    # Main training script (with progress tracking)
├── train.ipynb                 # Jupyter notebook for Google Colab
├── play.py                     # Script to watch trained agent play
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── env/                        # Virtual environment (local only)
│   ├── bin/
│   ├── lib/
│   └── ...
│
├── models/                     # Trained models (created after training)
│   ├── experiment_1/
│   │   ├── best_model.zip              # Best performing model
│   │   ├── final_model.zip             # Model at end of training
│   │   ├── checkpoint_50000_steps.zip  # Checkpoint at 50k steps
│   │   ├── checkpoint_100000_steps.zip # Checkpoint at 100k steps
│   │   ├── training_results.json       # Results in JSON format
│   │   └── training_results.txt        # Human-readable results
│   ├── experiment_2/
│   │   └── ...
│   ├── experiment_3/
│   │   └── ...
│   └── experiments_comparison.txt      # Comparison of all experiments
│
└── logs/                       # Training logs (created after training)
    ├── experiment_1/
    │   ├── evaluations.npz             # Evaluation metrics
    │   └── DQN_1/                      # TensorBoard logs
    │       └── events.out.tfevents.*
    ├── experiment_2/
    │   └── ...
    └── experiment_3/
        └── ...
```

---

## Technologies Used

| Technology            | Version | Purpose                     |
| --------------------- | ------- | --------------------------- |
| **Python**            | 3.9+    | Programming language        |
| **Stable Baselines3** | 2.1.0   | DQN implementation          |
| **Gymnasium**         | 0.29.1  | Atari environment framework |
| **ALE-Py**            | 0.8.1   | Arcade Learning Environment |
| **TensorBoard**       | 2.14.0  | Training visualization      |
| **NumPy**             | 1.24.3  | Numerical computations      |
| **OpenCV**            | 4.8.1   | Image processing            |

---

## Installation

### Local Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/pimanzi/Breakout-v5-reinforcement-learning.git
cd Breakout-v5-reinforcement-learning
```

#### 2. Create Virtual Environment

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Install Atari ROMs

```bash
AutoROM --accept-license
```

#### 5. Verify Installation

```bash
python -c "import gymnasium as gym; import ale_py; from stable_baselines3 import DQN; print(' Installation successful!')"
```

### Google Colab Setup

#### 1. Upload `train.ipynb` to Google Colab

#### 2. Install Dependencies (first cell)

```python
!pip install stable-baselines3[extra]==2.1.0 gymnasium==0.29.1 ale-py==0.8.1 autorom[accept-rom-license] tensorboard shimmy[atari]
```

#### 3. Import and Register ALE

```python
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
```

#### 4. Start Training!

```python
# Change experiment configuration in the notebook
EXPERIMENT_TO_RUN = "experiment_1"
TOTAL_TIMESTEPS = 500000
```

---

## How to Use

### Training a Model

#### Method 1: Using train.py (Local/Terminal)

**Quick Start:**

```bash
# Activate virtual environment
source env/bin/activate

# Train baseline CNN experiment
python train.py
```

**Configuration:**
Open `train.py` and modify lines 457-460:

```python
EXPERIMENT_TO_TRAIN = "experiment_1"  # Change to experiment_1 through experiment_10
TOTAL_TIMESTEPS = 500000              # 500k recommended, or 1000000 for best results
SAVE_DIR = "./models"
LOG_DIR = "./logs"
```

**Command-line Options:**

```bash
# Train specific experiment
python train.py --experiment experiment_2

# Train with different timesteps
python train.py --timesteps 1000000

# Generate comparison report after training multiple experiments
python train.py --compare
```

#### Method 2: Using train.ipynb (Google Colab)

1. Open `train.ipynb` in Google Colab
2. Run the installation cell
3. Modify the configuration cell:
   ```python
   EXPERIMENT_TO_RUN = "experiment_1"
   TOTAL_TIMESTEPS = 500000
   ```
4. Run the training cell
5. Use TensorBoard cell to monitor progress

**Expected Training Times:**

| Timesteps | Hardware       | Time        | Expected Reward |
| --------- | -------------- | ----------- | --------------- |
| 100k      | CPU            | 1-2 hours   | 5-15            |
| 100k      | GPU (Colab T4) | 30-60 min   | 5-15            |
| 500k      | CPU            | 5-10 hours  | 20-35           |
| 500k      | GPU (Colab T4) | 2-4 hours   | 20-35           |
| 1M        | CPU            | 10-20 hours | 30-50           |
| 1M        | GPU (Colab T4) | 4-8 hours   | 30-50           |

---

### Monitoring Training Progress

#### 1. Console Output

During training, you'll see progress updates every 5,000 steps:

```
================================================================================
Step: 50,000 / 500,000
[████████------------------------------------] 10.0%
Speed: 285 steps/sec
Elapsed: 0:02:55
ETA: 0:26:15
================================================================================
```

You'll also see metrics every 100 steps:

```
| rollout/ep_rew_mean      | 12.5    |  ← Average reward
| rollout/ep_len_mean      | 245     |  ← Average episode length
| rollout/exploration_rate | 0.95    |  ← Epsilon value
| time/total_timesteps     | 50000   |  ← Progress
```

#### 2. TensorBoard (Recommended!)

**Start TensorBoard:**

```bash
# In a new terminal
source env/bin/activate
tensorboard --logdir ./logs
```

**Open in browser:**

```
http://localhost:6006
```

**Key Metrics to Watch:**

- `eval/mean_reward` - Average reward during evaluation (most important!)
- `rollout/ep_rew_mean` - Average reward during training
- `train/loss` - Training loss (should decrease)
- `rollout/exploration_rate` - Epsilon decay

**In Google Colab:**

```python
%load_ext tensorboard
%tensorboard --logdir ./logs
```

#### 3. Evaluation Output

Every 10,000 steps, automatic evaluation runs:

```
Eval num_timesteps=10000, episode_reward=8.40 +/- 2.50
Episode length: 245.00 +/- 45.00
New best mean reward!
```

---

### Testing a Trained Model

#### Watch Your Agent Play:

```bash
# Activate virtual environment
source env/bin/activate

# Test the best model
python play.py --model ./models/experiment_1/best_model.zip

# Test with more episodes
python play.py --model ./models/experiment_1/best_model.zip --episodes 10

# Test without rendering (faster)
python play.py --model ./models/experiment_1/best_model.zip --no-render
```

**What You'll See:**

- Live game window showing agent playing
- Episode rewards printed to console
- Average reward across all episodes

---

## File Locations

### Where to Find Everything:

#### 📦 **Trained Models**

```
./models/experiment_X/
├── best_model.zip              ← USE THIS ONE! (Best performing model)
├── final_model.zip             ← Model at end of training
├── checkpoint_50000_steps.zip  ← Checkpoints for recovery
├── checkpoint_100000_steps.zip
├── training_results.json       ← Results in JSON format
└── training_results.txt        ← Human-readable results
```

**Important:** Always use `best_model.zip` for testing/evaluation!

#### **Training Logs**

```
./logs/experiment_X/
├── evaluations.npz             ← Numpy file with all evaluation rewards
└── DQN_1/
    └── events.out.tfevents.*   ← TensorBoard event files
```

**Load Evaluation Data:**

```python
import numpy as np
data = np.load('./logs/experiment_1/evaluations.npz')
rewards = data['results']  # Array of mean rewards
print(f"Best reward: {np.max(rewards):.2f}")
```

#### **Results Summary**

```
./models/experiment_X/training_results.txt
```

Contains:

- Hyperparameters used
- Best mean reward achieved
- Training duration
- File locations
- Next steps

#### **Comparison Report**

```
./models/experiments_comparison.txt
```

Generated after training multiple experiments:

```bash
python train.py --compare
```

Ranks all experiments by performance!

---

## Experiments

This project includes **10 different experiments** to test various hyperparameters and architectures:

### Experiments Overview:

| Experiment        | Name                   | Key Difference            | Policy        | Purpose                       |
| ----------------- | ---------------------- | ------------------------- | ------------- | ----------------------------- |
| **experiment_1**  | Baseline CNN           | Standard parameters       | CnnPolicy     | Baseline comparison           |
| **experiment_2**  | Higher Learning Rate   | lr=5e-4 (5x baseline)     | CnnPolicy     | Test faster learning          |
| **experiment_3**  | Lower Learning Rate    | lr=5e-5 (0.5x baseline)   | CnnPolicy     | Test slower, stable learning  |
| **experiment_4**  | Larger Buffer          | buffer=500k (5x baseline) | CnnPolicy     | Test more diverse experiences |
| **experiment_5**  | Larger Batch           | batch=64 (2x baseline)    | CnnPolicy     | Test gradient stability       |
| **experiment_6**  | Higher Gamma           | gamma=0.995 (vs 0.99)     | CnnPolicy     | Test long-term planning       |
| **experiment_7**  | More Exploration       | explore=0.2 (2x baseline) | CnnPolicy     | Test exploration impact       |
| **experiment_8**  | Frequent Target Update | target_update=500         | CnnPolicy     | Test learning stability       |
| **experiment_9**  | Early Training Start   | learning_starts=5000      | CnnPolicy     | Test early training           |
| **experiment_10** | MLP Architecture       | Same as baseline          | **MlpPolicy** | Compare architectures         |

### Baseline Hyperparameters (Experiment 1):

```python
learning_rate = 1e-4
buffer_size = 100,000
batch_size = 32
gamma = 0.99
exploration_fraction = 0.1
exploration_final_eps = 0.01
target_update_interval = 1000
learning_starts = 10,000
train_freq = 4
policy = CnnPolicy
```

---

## Troubleshooting

### Common Issues:

#### 1. `ModuleNotFoundError: No module named 'ale_py'`

**Solution:**

```bash
pip install ale-py==0.8.1
AutoROM --accept-license
```

#### 2. `Namespace ALE not found`

**Solution:** Make sure to import and register ALE:

```python
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
```

#### 3. Training is very slow on CPU

**Solution:**

- Use Google Colab with GPU (free)
- Reduce timesteps for testing: `TOTAL_TIMESTEPS = 100000`
- Use smaller experiments first

#### 4. Out of memory error

**Solution:**

- Reduce buffer size: `buffer_size = 50000`
- Reduce batch size: `batch_size = 16`
- Close other applications

#### 5. TensorBoard not showing graphs

**Solution:**

```bash
# Make sure logs directory exists and has data
ls ./logs/experiment_1/

# Restart TensorBoard with correct path
tensorboard --logdir ./logs --reload_interval 5
```

#### 6. Can't see the game window when using play.py

**Solution:**

- Make sure you're not using `--no-render` flag
- On remote servers (SSH), rendering won't work
- For Colab, use `render_mode='rgb_array'` and display frames manually

---

## License

This project is open source and available for educational purposes.

---

## Acknowledgments

- **Stable Baselines3** - For the DQN implementation
- **Gymnasium** - For the Atari environment framework
- **DeepMind** - For the original DQN paper (Mnih et al., 2015)

---

## Contact

**Repository:** [https://github.com/pimanzi/Breakout-v5-reinforcement-learning](https://github.com/pimanzi/Breakout-v5-reinforcement-learning)

**Author:** pimanzi

---

_Last Updated: November 17, 2025_
