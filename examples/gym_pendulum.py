import logging
import math
import time

import gymnasium as gym
import numpy as np
import torch
import torch.autograd
from gymnasium import wrappers
from gymnasium import logger as gym_log
from mpc import mpc

# gym_log.min_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 10  # T
    N_BATCH = 1
    LQR_ITER = 5
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0
    dtype = torch.float32

    class PendulumDynamics(torch.nn.Module):
        def forward(self, state, action):
            th = state[:, 0].view(-1, 1)
            thdot = state[:, 1].view(-1, 1)

            g = 10
            m = 1
            l = 1
            dt = 0.05

            u = action
            u = torch.clamp(u, -2, 2)

            newthdot = (
                thdot
                + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3.0 / (m * l**2) * u) * dt
            )
            newth = th + newthdot * dt
            newthdot = torch.clamp(newthdot, -8, 8)

            state = torch.cat((angle_normalize(newth), newthdot), dim=1)
            return state

    def angle_normalize(x):
        return ((x + math.pi) % (2 * math.pi)) - math.pi

    downward_start = True
    env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=None)
    env = wrappers.RecordVideo(
        env,
        video_folder="videos/pendulum_mpc",
        episode_trigger=lambda ep_id: True,
    )
    env = wrappers.RecordEpisodeStatistics(env)
    obs, info = env.reset()
    if downward_start:
        env.unwrapped.state = [np.pi, 1]
        obs = np.array(env.unwrapped.state, dtype=np.float64)

    nx = 2
    nu = 1

    u_init = None
    render = True
    retrain_after_iter = 50
    run_iter = 500

    # swingup goal (observe theta and theta_dt)
    goal_weights = torch.tensor((1.0, 0.1), dtype=dtype)  # nx
    goal_state = torch.tensor((0.0, 0.0), dtype=dtype)  # nx
    ctrl_penalty = 0.001
    q = torch.cat((goal_weights, ctrl_penalty * torch.ones(nu, dtype=dtype)))  # nx + nu
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(nu, dtype=dtype)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

    # run MPC
    total_reward = 0
    for i in range(run_iter):
        state = torch.tensor(env.unwrapped.state, dtype=dtype).view(1, -1)
        command_start = time.perf_counter()
        # recreate controller using updated u_init (kind of wasteful right?)
        ctrl = mpc.MPC(
            nx,
            nu,
            TIMESTEPS,
            u_lower=ACTION_LOW,
            u_upper=ACTION_HIGH,
            lqr_iter=LQR_ITER,
            exit_unconverged=False,
            eps=1e-2,
            n_batch=N_BATCH,
            backprop=False,
            verbose=0,
            u_init=u_init,
            grad_method=mpc.GradMethods.AUTO_DIFF,
        )

        # compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = ctrl(
            state, cost, PendulumDynamics()
        )
        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat(
            (nominal_actions[1:], torch.zeros(1, N_BATCH, nu, dtype=dtype)), dim=0
        )

        elapsed = time.perf_counter() - command_start
        np_action = action.detach().cpu().numpy().reshape(-1)
        obs, r, terminated, truncated, info = env.step(np_action)
        total_reward += r
        logger.debug(
            "action taken: %.4f cost received: %.4f time taken: %.5fs",
            action,
            -r,
            elapsed,
        )
        if render:
            env.render()

        if terminated or truncated:
            obs, info = env.reset()
            if downward_start:
                env.unwrapped.state = [np.pi, 1]
                obs = np.array(env.unwrapped.state, dtype=np.float32)

    logger.info("Total reward %f", total_reward)
