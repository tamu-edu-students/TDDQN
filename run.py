import os
import argparse
import csv
from time import time
from gym.wrappers.time_limit import TimeLimit
from gym_gridverse.gym import GymEnvironment
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import make_observation_representation
from gym_gridverse.representations.state_representations import make_state_representation
from envs.gv_wrapper import GridVerseWrapper
import os
import torch
from tqdm import tqdm
import random
import numpy as np
from tddqn.agents.dqn import DqnAgent
from tddqn.agents.drqn import DrqnAgent
from tddqn.agents.dtqn import DtqnAgent
from tddqn.agents.tddqn import TddqnAgent
from tddqn.networks.drqn import DRQN
from tddqn.networks.dqn import DQN
from tddqn.networks.dtqn import DTQN
from tddqn.networks.tddqn import TDDQN



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="DTQN")
    parser.add_argument("--env",type=str,default="custom_env.yaml")
    parser.add_argument("--num-steps",type=int,default=50_000)
    parser.add_argument("--tuf",type=int,default=10_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--buf-size",type=int,default=500_000)
    parser.add_argument("--eval-frequency",type=int,default=5_000)
    parser.add_argument("--eval-episodes",type=int,default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context",type=int,default=50)
    parser.add_argument("--obs-embed",type=int,default=8)
    parser.add_argument("--a-embed",type=int,default=0)
    parser.add_argument("--in-embed",type=int,default=128)
    parser.add_argument("--max-episode-steps",type=int,default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-policy",action="store_true")
    parser.add_argument("--verbose",default = True, action="store_true")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--history",type=int,default=50)
    # TDDQN-Specific
    parser.add_argument("--heads",type=int,default=4,)
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--gate",type=str,default="res")
    parser.add_argument("--identity",action="store_true")
    parser.add_argument("--pos", default="learned")
    parser.add_argument("--bag-size", type=int, default=0)
    return parser.parse_args()


def make_env(env):
    print("Loading Environment")
    env_path =  os.path.join(os.getcwd(), "envs", env)
    inner_env = factory_env_from_yaml(env_path)
    state_representation = make_state_representation("default", inner_env.state_space)
    observation_representation = make_observation_representation("default", inner_env.observation_space)
    outer_env = OuterEnv(inner_env,state_representation=state_representation,observation_representation=observation_representation)
    env = GymEnvironment(outer_env)
    env = TimeLimit(GridVerseWrapper(env), max_episode_steps=250)
    return env

def get_agent(model_str, env, embed_per_obs_dim, action_dim, inner_embed, buffer_size, device, learning_rate, batch_size, context_len, max_env_steps, history, target_update_frequency, gamma, num_heads=1, num_layers=1, dropout=0.0, identity=False, gate="res", pos="learned", bag_size=0):
    agent_classes = {
        "DTQN": (DtqnAgent, DTQN),
        "DQN": (DqnAgent, DQN),
        "DRQN": (DrqnAgent, DRQN),
        "TDDQN": (TddqnAgent, TDDQN)
    }
    agent_class, network_creator = agent_classes[model_str]

    env_obs_length = env.observation_space.shape[0]
    env_obs_mask = max(env.observation_space.nvec) + 1
    max_env_steps = env._max_episode_steps
    obs_vocab_size = env_obs_mask + 1
    history = min(max(history, 1), context_len)
    num_actions = env.action_space.n

    if model_str == "DQN":
        context_len = 1
        network = lambda: network_creator(
        env_obs_length, num_actions, embed_per_obs_dim, action_dim, inner_embed, obs_vocab_size).to(device)
    elif model_str == "DRQN":
        network = lambda: network_creator(
        env_obs_length, num_actions, embed_per_obs_dim, action_dim, inner_embed, obs_vocab_size, batch_size).to(device)
    else:
        network = lambda: network_creator(
            env_obs_length, num_actions, embed_per_obs_dim, action_dim, inner_embed, num_heads, num_layers,
            context_len, dropout=dropout, gate=gate, identity=identity, pos=pos, vocab_sizes=obs_vocab_size,
            target_update_frequency=target_update_frequency, bag_size=bag_size).to(device)

    return agent_class(
        network, buffer_size, device, env_obs_length, max_env_steps, env_obs_mask, num_actions,
        learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, context_len=context_len,
        embed_size=inner_embed, history=history, target_update_frequency=target_update_frequency,
        bag_size=bag_size
    )


def linear_anneal(start, end, duration, step):
    return max(end, start - (start - end) * step / duration)

def evaluate(agent, eval_env, eval_episodes, render = False):
    agent.eval_on()
    total_reward = 0
    num_successes = 0
    total_steps = 0

    for _ in range(eval_episodes):
        state = eval_env.reset()
        agent.context_reset(state)
        episode_done = False
        episode_reward = 0.0

        while not episode_done:
            action = agent.get_action(epsilon=0.0)  # Greedy policy
            next_state, reward, episode_done, info = eval_env.step(action)
            agent.observe(next_state, action, reward, episode_done)
            episode_reward += reward

            if render:
                eval_env.render()
                time.sleep(0.1)  # Reduce render speed for better visibility

        total_reward += episode_reward
        total_steps += agent.context.timestep

        # Define success criteria; customize based on specific needs or environment
        if info.get("is_success", False) or episode_reward > 0:
            num_successes += 1

    # Set networks back to train mode
    agent.eval_off()
    mean_success = num_successes / eval_episodes if eval_episodes > 0 else 0
    mean_reward = total_reward / eval_episodes if eval_episodes > 0 else 0
    mean_episode_length = total_steps / eval_episodes if eval_episodes > 0 else 0

    return mean_success, mean_reward, mean_episode_length


def train(agent, env, total_steps, eps_val, eval_frequency, eval_episodes, policy_path, save_policy, num_steps, verbose= False):
    start_time = time()
    eps_start = 1.0
    eps_end = 0.1
    eps_duration = num_steps // 10
    results_path = policy_path + "_results.csv"
    # Turn on train mode
    agent.eval_off()
    agent.context_reset(env.reset())
    for timestep in tqdm(range(agent.num_train_steps, total_steps)):
        done = step(agent, env, eps_val)
        if done:
            agent.replay_buffer.flush()
            agent.context_reset(env.reset())
        agent.train()
        eps_val = linear_anneal(eps_start, eps_end, eps_duration, timestep)
        if timestep % eval_frequency == 0:
            hours = (time() - start_time) / 3600
            sr, ret, length = evaluate(agent, env, eval_episodes)
            with open(results_path, "a") as file:
                csv.writer(file).writerow([sr, ret, length])
            if verbose:
                print(f"Training Steps: {timestep:,}, " f"Success Rate: {sr:.2%}, " f"Return: {ret:.2f}, " f"Episode Length: {length:.2f}, "f"Hours: {hours:.2f}")
            if save_policy:
                torch.save(agent.policy_network.state_dict(), policy_path)


def step(agent, env, eps_val):
    action = agent.get_action(epsilon=eps_val)
    next_obs, reward, done, info = env.step(action)
    if info.get("TimeLimit.truncated", False):
        buffer_done = False
    else:
        buffer_done = done
    agent.observe(next_obs, action, reward, buffer_done)
    return done


def prepopulate(agent, prepop_steps, env):
    timestep = 0
    while timestep < prepop_steps:
        agent.context_reset(env.reset())
        done = False
        while not done:
            action = np.random.default_rng().integers(env.action_space.n)
            next_obs, reward, done, info = env.step(action)
            if info.get("TimeLimit.truncated", False):
                buffer_done = False
            else:
                buffer_done = done
            agent.observe(next_obs, action, reward, buffer_done)
            timestep += 1
        agent.replay_buffer.flush()


def run_experiment(args):
    # Create envs, set seed, create RL agent
    env = make_env(args.env)
    device = torch.device(args.device)
    # Fix the seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # learning rate params
    eps = 1
    agent = get_agent(args.model, env, args.obs_embed, args.a_embed, args.in_embed, args.buf_size, device, args.lr, args.batch, args.context, args.max_episode_steps, args.history, args.tuf, args.discount, args.heads, args.layers, args.dropout, args.identity, args.gate, args.pos, args.bag_size)
    print(f"Creating {args.model} with {sum(p.numel() for p in agent.policy_network.parameters()):,} parameters")
    policy_save_dir = os.path.join( os.getcwd(), "policies", args.env)
    os.makedirs(policy_save_dir, exist_ok=True)
    policy_path = os.path.join(policy_save_dir, f"model={args.model}_seed={args.seed}")
    # Prepopulate the replay buffer
    prepopulate(agent, 50_000, env)
    train(agent, env, args.num_steps, eps, args.eval_frequency, args.eval_episodes, policy_path, args.save_policy, args.num_steps, args.verbose)


if __name__ == "__main__":
    run_experiment(get_args())
