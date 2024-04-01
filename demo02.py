import numpy as np
import torch
import os
import argparse
from omegaconf import OmegaConf
import plotly.graph_objects as go
from envs import get_env
from envs.models.panda.panda import Panda
from envs.lib.LieGroup import invSE3
def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    if (val == 'True') or (val == 'true'):
        return True
    if (val == 'False') or (val == 'false'):
        return False
    try:
        return float(val)
    except:
        return str(val)


def parse_unknown_args(l_args):
    """convert the list of unknown args into dict
    this does similar stuff to OmegaConf.from_cli()
    I may have invented the wheel again..."""
    n_args = len(l_args) // 2
    kwargs = {}
    for i_args in range(n_args):
        key = l_args[i_args * 2]
        val = l_args[i_args * 2 + 1]
        assert '=' not in key, 'optional arguments should be separated by space'
        kwargs[key.strip('-')] = parse_arg_type(val)
    return kwargs


def parse_nested_args(d_cmd_cfg):
    """produce a nested dictionary by parsing dot-separated keys
    e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split('.')
        d = d_new_cfg
        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}
                d = d[each_key]
    return d_new_cfg
def compute_min_distance_and_visualize(cfg):
    # Load configuration
    n_data = cfg.n_data
    n_pcd = cfg.n_pcd

    n_radius = cfg.n_radius
    radius_range = cfg.radius_range

    n_theta = cfg.n_theta
    theta_range = cfg.theta_range

    n_phi = cfg.n_phi
    phi_range = cfg.phi_range

    n_env = n_radius * n_theta * n_phi

    n_data_per_env = int(np.ceil(n_data / n_env))

    # Initialize lists to store results

    distances_list = [None] * n_env

    env_idx = 0

    # Loop over different environments
    for radius in np.linspace(*radius_range, n_radius):
        for theta in np.linspace(*theta_range, n_theta):
            for phi in np.linspace(*phi_range, n_phi):

                # Create environment configuration
                env_cfg = {
                    'name': 'multipanda',
                    'base_poses': [
                        [0, 0, 0],
                        [radius * np.cos(theta), radius * np.sin(theta), 0]
                    ],
                    'base_orientations': [0, phi]
                }

                # Get environment
                env = get_env(env_cfg=env_cfg)

                # Get link pairs for calculating distances
                link_pairs = []
                links = list(range(env.n_dof)) + [(x + 1) * (-1) for x in range(env.n_robot)]
                for i_idx in links:
                    for j_idx in [x for x in links if x != i_idx]:
                        link_pairs.append([i_idx, j_idx])

                link_pairs = torch.tensor(link_pairs)

                # Generate random joint configurations
                n_sample = int(np.ceil(n_data_per_env / len(link_pairs)))
                data_q = torch.rand(n_sample, env.n_dof) * (torch.tensor(env.q_max) - torch.tensor(env.q_min)) + torch.tensor(env.q_min)

                # Calculate distances
                distances = env.calculate_distance_between_links(data_q, link_pairs, pbar=False)



    # Concatenate results
    distances = torch.cat(distances_list, dim=0)


    # Visualize distances
    visualize_distances(distances)

def visualize_distances(distances):
    # Assuming distances is a torch tensor of shape (N,)
    # Plotting histogram of distances
    fig = go.Figure(data=[go.Histogram(x=distances.numpy())])
    fig.update_layout(title="Distribution of pairwise collision distances", xaxis_title="Distance", yaxis_title="Frequency")
    fig.show()
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)

    # 加载配置文件
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)

    # 运行计算最小距离并可视化函数
    compute_min_distance_and_visualize(cfg)
