import numpy as np
import os
import sys
sys.path.append(r'C:\Users\a1fla\Desktop\MagDrone')
import torch
import torch.nn as nn 
from datetime import datetime
from trim.trim_hrz import FixedWingDynamics, TrimCalculator, WindField
from stable_baselines3 import TD3  # Use TD3 instead of SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from env.WindCompEnv import WindCompEnv

# Set random seeds
#np.random.seed(42)
#torch.manual_seed(42)

# Train TD3 to optimize dual PID parameters for wind compensation
def train_td3_for_compensation(env_class, offline=False, 
                              total_timesteps=10_0000,
                              use_pretrained=True, 
                              pretrained_model_path=None,
                              vec_normalize_path=None):
    # Fix: Create environment constructor
    def make_env(env_id=0):
        """Create independent environment instance"""
        env = env_class()  # Call the passed environment constructor
        env.seed(env_id)   # Set different random seeds for each environment
        return env
    
    # Create SB3 compatible vectorized environment
    vec_env = make_vec_env(make_env, n_envs=5)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Load existing VecNormalize statistics if available
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = True  # Keep training mode to allow updating statistics during continued training
        vec_env.norm_reward = True  # Keep reward normalization enabled (consistent with previous training)
    else:
        pass
    
    # Independent evaluation environment
    def make_eval_env(env_id=0):
        """Create evaluation environment with separate random seeds"""
        env = env_class()
        env.seed(env_id + 100)  # Use different random seeds from training environments
        return env
    
    eval_vec_env = make_vec_env(make_eval_env, n_envs=1)  # Evaluation typically uses 1 environment
    # ✅ Key: Evaluation environment needs the same VecNormalize settings but uses training stats
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=True)
    eval_vec_env.training = False  # Do not update statistics during evaluation
    
    # Sync statistics from training environment when loading from checkpoint
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        eval_vec_env = VecNormalize.load(vec_normalize_path, eval_vec_env)
        eval_vec_env.training = False  # Keep statistics fixed for evaluation
        eval_vec_env.norm_reward = True
    else:
        eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=True)
        eval_vec_env.training = False

    if offline:
        pass
    else:
        print("Starting online training: (TD3)")
        print(f"Number of parallel environments created: {vec_env.num_envs}")
        
        # Define policy network parameters
        policy_kwargs = {
            'net_arch': [256, 128, 64, 32],
            'activation_fn': nn.ReLU,
        }

        # Create action noise
        n_actions = vec_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.005 * np.ones(n_actions))

        # Load pretrained model or create new TD3 agent
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pretrained model: {pretrained_model_path}")
            td3_agent = TD3.load(
                pretrained_model_path,
                env=vec_env,
                tensorboard_log=f"../logs/td3/MagDrone_td3_tensorboard/td3_{datetime.now().strftime('%Y%m%d_%H%M')}",
                device='auto',
                learning_rate=5e-5,
                buffer_size=500_000,
                learning_starts=10000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                action_noise=action_noise,  # TD3-specific action noise
                policy_delay=8,  # TD3-specific: policy update delay
                target_policy_noise=0.1,  # TD3-specific: target policy noise
                target_noise_clip=0.5,  # TD3-specific: noise clipping
                policy_kwargs=policy_kwargs
            )
            print(f"Pretrained model parameters loaded. Will continue training for {total_timesteps} steps from checkpoint.")
        else:
            # Initialize new TD3 agent
            td3_agent = TD3(
                "MlpPolicy", 
                vec_env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=100_0000,
                learning_starts=10000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,  # TD3-specific action noise
                policy_delay=2,  # TD3-specific: policy update delay
                target_policy_noise=0.2,  # TD3-specific: target policy noise
                target_noise_clip=0.5,  # TD3-specific: noise clipping
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"../logs/td3/MagDrone_td3_tensorboard/td3_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )

        # Callbacks configuration
        # Evaluation callback: save best model based on evaluation performance
        eval_callback = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"../logs/td3/MagDrone_logs/best_model_{datetime.now().strftime('%Y%m%d_%H%M')}",
            log_path='../logs/td3/MagDrone_logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Checkpoint callback: save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=1_0000,
            save_path=f"../logs/td3/MagDrone_logs/checkpoints/run_{datetime.now().strftime('%Y%m%d_%H%M')}",
            name_prefix="MagDrone_rl_model"
        )

        # Train the TD3 model
        td3_agent.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False
        )

        # Save final model and VecNormalize statistics
        model_path = f"../logs/td3/MagDrone_models/td3_MagDrone_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        vec_path = f"../logs/td3/MagDrone_models/vec_normalize_stats_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        os.makedirs('../logs/td3/MagDrone_models', exist_ok=True)
        td3_agent.save(model_path)
        vec_env.save(vec_path)
        print(f"Training completed. Model saved to: {model_path}")
        print(f"VecNormalize statistics saved to: {vec_path}")
        
        return td3_agent, [], None

# Main training function
def main():
    # Define environment constructor
    def create_env():
        """Create WindCompEnv with fixed-wing dynamics, trim calculator and wind field"""
        wind_field = WindField()
        model = FixedWingDynamics(wind_field)
        trim_calc = TrimCalculator(model, wind_field)
        target_heading = np.deg2rad(10.0)
        return WindCompEnv(model, trim_calc, wind_field, target_heading)
    
    # Define TD3 model and VecNormalize paths
    td3_model_path = f"../logs/td3/MagDrone_models/td3_MagDrone_20251209_1909.zip"
    vec_normalize_path = f"../logs/td3/MagDrone_models/vec_normalize_stats_20251209_1909.pkl"
    
    print("Starting TD3 online training mode...")
    
    # Check for existing pretrained TD3 model
    use_existing_td3 = os.path.exists(td3_model_path)
    if use_existing_td3:
        print(f'Pretrained model exists. Training with {td3_model_path}!')
        td3_agent, _, _ = train_td3_for_compensation(
            create_env, offline=False, total_timesteps=320_0000,
            use_pretrained=use_existing_td3,
            pretrained_model_path=td3_model_path if use_existing_td3 else None,
            vec_normalize_path=vec_normalize_path
        )
    else:
        # Train new TD3 agent from scratch
        td3_agent, _, _ = train_td3_for_compensation(
            create_env, offline=False, total_timesteps=100_0000,
            use_pretrained=use_existing_td3,
            pretrained_model_path=td3_model_path if use_existing_td3 else None,
            vec_normalize_path=vec_normalize_path
        )
    
    print("Training finished!")

if __name__ == "__main__":
    main()