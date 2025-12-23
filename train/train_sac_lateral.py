import numpy as np
import os
import sys
sys.path.append(r'C:\Users\a1fla\Desktop\MagDrone')
import torch
import torch.nn as nn 
from datetime import datetime
from trim.trim_hrz import FixedWingDynamics, TrimCalculator, WindField
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env.WindCompEnv import WindCompEnv

# Set random seeds
#np.random.seed(42)
#torch.manual_seed(42)

# Train SAC to optimize dual PID parameters for wind compensation
def train_sac_for_compensation(env_class, offline=False, 
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
        print("Starting online training: (SAC)")
        print(f"Number of parallel environments created: {vec_env.num_envs}")
        
        # Policy network configuration
        policy_kwargs = {
            'net_arch': [256, 128, 64, 32],
            'activation_fn': nn.ReLU,
            "use_sde": False
        }

        # Load pretrained model or create new SAC agent
        if pretrained_model_path:
            print(f"Loading pretrained model: {pretrained_model_path}")
            sac_agent = SAC.load(
                pretrained_model_path,
                env=vec_env,
                tensorboard_log=f"../logs/sac/MagDrone_sac_tensorboard/sac_{datetime.now().strftime('%Y%m%d_%H%M')}",
                device='auto',
                learning_rate=1e-4,
                buffer_size=100_0000,
                learning_starts=10000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=4,
                ent_coef='auto',
                target_update_interval=1,
                policy_kwargs=policy_kwargs
            )
            #sac_agent.replay_buffer.reset() 
            print(f"Pretrained model parameters loaded. Will continue training for {total_timesteps} steps from checkpoint.")
        else:
            # Initialize new SAC agent
            sac_agent = SAC(
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
                ent_coef='auto',
                target_update_interval=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"../logs/sac/MagDrone_sac_tensorboard/sac_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )

        # Callbacks configuration
        # Evaluation callback: save best model based on evaluation performance
        eval_callback = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"../logs/sac/MagDrone_logs/best_model_{datetime.now().strftime('%Y%m%d_%H%M')}",
            log_path='../logs/sac/MagDrone_logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Checkpoint callback: save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=1_0000,
            save_path=f"../logs/sac/MagDrone_logs/checkpoints/run_{datetime.now().strftime('%Y%m%d_%H%M')}",
            name_prefix="MagDrone_rl_model"
        )

        # Train the SAC model
        sac_agent.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False
        )

        # Save final model and VecNormalize statistics
        model_path = f"../logs/sac/MagDrone_models/sac_MagDrone_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        vec_path = f"../logs/sac/MagDrone_models/vec_normalize_stats_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        os.makedirs('../logs/sac/MagDrone_models', exist_ok=True)
        sac_agent.save(model_path)
        vec_env.save(vec_path)
        print(f"Training completed. Model saved to: {model_path}")
        print(f"VecNormalize statistics saved to: {vec_path}")
        
        return sac_agent, [], None

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
    
    # Define SAC model and VecNormalize paths
    sac_model_path = f"../logs/sac/MagDrone_models/sac_MagDrone_20251206_1741.zip"
    vec_normalize_path = f"../logs/sac/MagDrone_models/vec_normalize_stats_20251206_1741.pkl"
    
    print("Starting SAC online training mode...")
    
    # Check for existing pretrained SAC model
    use_existing_sac = os.path.exists(sac_model_path)
    if use_existing_sac:
        print(f'Pretrained model exists. Training with {sac_model_path}!')
        sac_agent, _, _ = train_sac_for_compensation(
            create_env, offline=False, total_timesteps=50_0000,
            use_pretrained=use_existing_sac,
            pretrained_model_path=sac_model_path if use_existing_sac else None,
            vec_normalize_path=vec_normalize_path
        )
    else:
        # Train new SAC agent from scratch
        sac_agent, _, _ = train_sac_for_compensation(
            create_env, offline=False, total_timesteps=100_0000,
            use_pretrained=use_existing_sac,
            pretrained_model_path=sac_model_path if use_existing_sac else None,
            vec_normalize_path=vec_normalize_path
        )
    
    print("Training finished!")

if __name__ == "__main__":
    main()