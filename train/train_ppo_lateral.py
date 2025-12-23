import numpy as np
import os
import sys
sys.path.append(r'C:\Users\a1fla\Desktop\MagDrone')
import torch
import torch.nn as nn 
from datetime import datetime
from trim.trim_hrz import FixedWingDynamics, TrimCalculator, WindField
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env.WindCompEnv import WindCompEnv

# Set random seeds
#np.random.seed(42)
#torch.manual_seed(42)

# Train PPO to optimize dual PID parameters for wind compensation
def train_ppo_for_compensation(env_class, offline=False, 
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

    if offline:
        pass
    else:
        print("Starting online training: (PPO)")
        print(f"Number of parallel environments created: {vec_env.num_envs}")
        
        # Policy network configuration
        policy_kwargs = {
            'net_arch': [256, 128, 64, 32],
            'activation_fn': nn.ReLU,
            'log_std_init': -1.0,
        }

        # Load pretrained model or create new PPO agent
        if pretrained_model_path:
            print(f"Loading pretrained model: {pretrained_model_path}")
            ppo_agent = PPO.load(
                pretrained_model_path,
                env=vec_env,
                tensorboard_log=f"../logs/ppo/KiKd/MagDrone_ppo_tensorboard/ppo_{datetime.now().strftime('%Y%m%d_%H%M')}",
                device='auto',  # Automatically select GPU/CPU
                learning_rate=1e-5,  
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.003,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs
            )
            print(f"Pretrained model parameters loaded. Will continue training for {total_timesteps} steps from checkpoint.")
        else:
            # Initialize new PPO agent
            ppo_agent = PPO(
                "MlpPolicy", 
                vec_env,
                verbose=1,
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.05,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"../logs/ppo/KiKd/MagDrone_ppo_tensorboard/ppo_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
        # ==========================================

        # Callbacks configuration
        # Evaluation callback: save best model based on evaluation performance
        eval_callback = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"../logs/ppo/KiKd/MagDrone_logs/best_model_{datetime.now().strftime('%Y%m%d_%H%M')}",
            log_path='../logs/ppo/KiKd/MagDrone_logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Checkpoint callback: save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=1_0000,
            save_path=f"../logs/ppo/KiKd/MagDrone_logs/checkpoints/run_{datetime.now().strftime('%Y%m%d_%H%M')}",
            name_prefix="MagDrone_rl_model"
        )

        # Train the PPO model
        ppo_agent.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False  # Key parameter: do not reset timestep counter
        )

        # Save final model and VecNormalize statistics
        model_path = f"../logs/ppo/KiKd/MagDrone_models/ppo_MagDrone_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        vec_path = f"../logs/ppo/KiKd/MagDrone_models/vec_normalize_stats_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        os.makedirs('../logs/ppo/KiKd/MagDrone_models', exist_ok=True)
        
        ppo_agent.save(model_path)
        vec_env.save(vec_path)
        
        print(f"Training completed. Model saved to: {model_path}")
        print(f"VecNormalize statistics saved to: {vec_path}")
        
        return ppo_agent, [], None

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
    
    # Define PPO model and VecNormalize paths
    #ppo_model_path = f"../logs/ppo/MagDrone_logs/best_model_20251202_2044/best_model.zip"
    #vec_normalize_path=f"../logs/ppo/MagDrone_models/vec_normalize_stats_20251202_2053.pkl"
    #ppo_model_path = '../logs/ppo/MagDrone_models/ppo_MagDrone_20251209_2139.zip'
    #vec_normalize_path='../logs/ppo/MagDrone_models/vec_normalize_stats_20251209_2139.pkl'
    ppo_model_path = ''
    vec_normalize_path=''
    
    print("Starting PPO online training mode...")
    
    # Check for existing pretrained PPO model
    use_existing_ppo = os.path.exists(ppo_model_path) if ppo_model_path else False
    if use_existing_ppo:
        print(f'Pretrained model exists. Training with {ppo_model_path}!')
        ppo_agent, _, _ = train_ppo_for_compensation(
            create_env, offline=False, total_timesteps=50_0000,
            use_pretrained=use_existing_ppo,
            pretrained_model_path=ppo_model_path if use_existing_ppo else None,
            vec_normalize_path=vec_normalize_path
        )
    else:
        # Train new PPO agent from scratch
        ppo_agent, _, _ = train_ppo_for_compensation(
            create_env, offline=False, total_timesteps=100_0000,
            use_pretrained=use_existing_ppo,
            pretrained_model_path=ppo_model_path if use_existing_ppo else None
        )
    
    print("Training finished!")

if __name__ == "__main__":
    main()