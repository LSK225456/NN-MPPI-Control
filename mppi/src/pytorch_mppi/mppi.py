import logging
import time
import typing

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from arm_pytorch_utilities import handle_batch_input
from functorch import vmap

logger = logging.getLogger(__name__)


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


class SpecificActionSampler:
    def __init__(self):
        self.start_idx = 0
        self.end_idx = 0
        self.slice = slice(0, 0)

    def sample_trajectories(self, state, info):
        raise NotImplementedError

    def specific_dynamics(self, next_state, state, action, t):
        """Handle dynamics in a specific way for the specific action sampler; defaults to using default dynamics"""
        return next_state

    def register_sample_start_end(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.slice = slice(start_idx, end_idx)


class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dynamics, running_cost, nx, noise_sigma, num_samples=1000, horizon=15, device="cpu",
                 terminal_state_cost=None,
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init=None,
                 u_scale=1,
                 u_per_command=1,
                 step_dependent_dynamics=False,
                 rollout_samples=1,
                 rollout_var_cost=0,
                 rollout_var_discount=0.95,
                 sample_null_action=False,
                 specific_action_sampler: typing.Optional[SpecificActionSampler] = None,
                 noise_abs_cost=False):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for models that output a distribution)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param specific_action_sampler: Function to explicitly sample actions to use instead of sampling from noise from
            nominal trajectory, may output a number of action trajectories fewer than horizon
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_command = u_per_command
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)     # 正态分布生成器。MPPI靠它来生成成千上万条“瞎走”的随机轨迹。
        # T x nu control sequence
        self.U = U_init     # [T, nu] 的张量（T=预测步长，nu=控制维度）,它存储了当前认为最优的未来控制序列（均值）
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics       # 以后是我传入的 LSTM 模型
        self.running_cost = running_cost        # 代价函数
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.specific_action_sampler = specific_action_sampler
        self.noise_abs_cost = noise_abs_cost
        self.state = None
        self.info = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

    def get_params(self):
        return f"K={self.K} T={self.T} M={self.M} lambda={self.lambda_} noise_mu={self.noise_mu.cpu().numpy()} noise_sigma={self.noise_sigma.cpu().numpy()}".replace(
            "\n", ",")

    @handle_batch_input(n=2)
    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    @handle_batch_input(n=2)
    def _running_cost(self, state, u, t):
        return self.running_cost(state, u, t) if self.step_dependency else self.running_cost(state, u)

    def get_action_sequence(self):
        return self.U

    def shift_nominal_trajectory(self):
        """
        Shift the nominal trajectory forward one step
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

    def command(self, state, shift_nominal_trajectory=True, info=None):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :param shift_nominal_trajectory: Whether to roll the nominal trajectory forward one step. This should be True
            if the command is to be executed. If the nominal trajectory is to be refined then it should be False.
        :param info: Optional dictionary to store context information
        :returns action: (nu) best action
        主接口，处理时间窗口滑动：上一帧算出的最优控制序列self.U是针对t, t+1, ...的，现在时间过了1步，所以把self.U整体左移一位
        """
        self.info = info
        if shift_nominal_trajectory:
            self.shift_nominal_trajectory()     # 热启动：上一帧计算出的最优控制序列 self.U 整体左移一步

        return self._command(state)     # 核心主循环

    def _compute_weighting(self, cost_total):
        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        return self.omega

    def _command(self, state):      # 采样、预测、打分
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()       ### 从正态分布中采样噪声，计算所有采样轨迹的代价

        self._compute_weighting(cost_total)     # 计算权重，给低代价的轨迹分配高权重 (self.omega)
        
        # 更新最优控制序列，优控制 = 旧控制 + 加权后的噪声
        perturbations = torch.sum(self.omega.view(-1, 1, 1) * self.noise, dim=0)
        self.U = self.U + perturbations

        action = self.get_action_sequence()[:self.u_per_command]
        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]
        return action       # 返回 self.U 的第一个动作 action[0] 给机器人执行

    def change_horizon(self, horizon):
        if horizon < self.U.shape[0]:
            # truncate trajectory
            self.U = self.U[:horizon]
        elif horizon > self.U.shape[0]:
            # extend with u_init
            self.U = torch.cat((self.U, self.u_init.repeat(horizon - self.U.shape[0], 1)))
        self.T = horizon

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions):        ### 推演动力学并计算状态代价
        K, T, nu = perturbed_actions.shape                      # K=采样数, T=预测步长, nu=控制维
        assert nu == self.nu

        # 准备好 state (当前状态，复制 K 份) 和 cost_total (全0)
        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        # rollout action trajectory M times to estimate expected cost
        state = state.repeat(self.M, 1, 1)

        states = []
        actions = []

        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)         # 取出第 t 时刻所有 K 个样本的控制量

            next_state = self._dynamics(state, u, t)    # 预测下一时刻状态，lstm模型接口处

            # potentially handle dynamics in a specific way for the specific action sampler可能需要为特定动作采样器，以一种定制化的方式处理动力学
            next_state = self._sample_specific_dynamics(next_state, state, u, t)

            state = next_state  # 更新状态

            c = self._running_cost(state, u, t) # 计算当前步代价，调用传入的 running_cost 函数，检查这个新状态 state 是否撞墙，返回代价 c

            cost_samples = cost_samples + c # 累加代价
            if self.M > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

            # Save total states/actions
            states.append(state)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples = cost_samples + c
        cost_total = cost_total + cost_samples.mean(dim=0)
        cost_total = cost_total + cost_var * self.rollout_var_cost
        return cost_total, states, actions

    def _compute_perturbed_action_and_noise(self):
        # parallelize sampling across trajectories，生成 K 条扰动控制序列
        # resample noise each time we take an action，它不是只生成一个控制量，而是生成了 K=500 套完全不同的未来驾驶计划。有的计划是猛踩油门，有的计划是急转弯。
        
        noise = self.noise_dist.rsample((self.K, self.T))   
        # 一次性采样 K 条轨迹、每条 T 步的噪声矩阵，noise shape: [K, T, nu] (K=采样数500, T=步长, nu=控制维)

        perturbed_action = self.U + noise            # 扰动控制序列 = 均值(U) + 噪声
        perturbed_action = self._sample_specific_actions(perturbed_action)

        self.perturbed_action = self._bound_action(perturbed_action)   # 执行限幅（Clip），确保控制量不超出电机物理极限

        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U

    def _sample_specific_actions(self, perturbed_action):
        # specific sampling of actions (encoding trajectory prior and domain knowledge to create biases)
        i = 0
        if self.sample_null_action:
            perturbed_action[i] = 0
            i += 1
        if self.specific_action_sampler is not None:
            actions = self.specific_action_sampler.sample_trajectories(self.state, self.info)
            # check how long it is
            actions = actions.reshape(-1, self.T, self.nu)
            perturbed_action[i:i + actions.shape[0]] = actions
            self.specific_action_sampler.register_sample_start_end(i, i + actions.shape[0])
            i += actions.shape[0]
        return perturbed_action

    def _sample_specific_dynamics(self, next_state, state, u, t):
        if self.specific_action_sampler is not None:
            next_state = self.specific_action_sampler.specific_dynamics(next_state, state, u, t)
        return next_state

    def _compute_total_cost_batch(self):
        self._compute_perturbed_action_and_noise()  # 利用正态分布生成 K 条并行的控制序列，存入 self.perturbed_action（维度 [K, T, nu]）
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv   # 计算控制本身的代价，惩罚过大的控制输入或与均值的偏差（平滑性）
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv  # Like original paper

        rollout_cost, self.states, actions = self._compute_rollout_costs(self.perturbed_action) # 推演动力学并计算状态代价
        self.actions = actions / self.u_scale

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            return torch.max(torch.min(action, self.u_max), self.u_min)
        return action

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def get_rollouts(self, state, num_rollouts=1, U=None):
        """
            :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
            :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                                 dynamics
            :returns states: num_rollouts x T x nx vector of trajectories

        """
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        if U is None:
            U = self.get_action_sequence()
        T = U.shape[0]
        states = torch.zeros((num_rollouts, T + 1, self.nx), dtype=U.dtype, device=U.device)
        states[:, 0] = state
        for t in range(T):
            next_state = self._dynamics(states[:, t].view(num_rollouts, -1),
                                        self.u_scale * U[t].tile(num_rollouts, 1), t)
            # dynamics may augment state; here we just take the first nx dimensions
            states[:, t + 1] = next_state[:, :self.nx]

        return states[:, 1:]


class SMPPI(MPPI):
    """Smooth MPPI by lifting the control space and penalizing the change in action from
    https://arxiv.org/pdf/2112.09988
    """

    def __init__(self, *args, w_action_seq_cost=1., delta_t=1., U_init=None, action_min=None, action_max=None,
                 **kwargs):
        self.w_action_seq_cost = w_action_seq_cost
        self.delta_t = delta_t

        super().__init__(*args, U_init=U_init, **kwargs)

        # these are the actual commanded actions, which is now no longer directly sampled
        self.action_min = action_min
        self.action_max = action_max
        if self.action_min is not None and self.action_max is None:
            if not torch.is_tensor(self.action_min):
                self.action_min = torch.tensor(self.action_min)
            self.action_max = -self.action_min
        if self.action_max is not None and self.action_min is None:
            if not torch.is_tensor(self.action_max):
                self.action_max = torch.tensor(self.action_max)
            self.action_min = -self.action_max
        if self.action_min is not None:
            self.action_min = self.action_min.to(device=self.d)
            self.action_max = self.action_max.to(device=self.d)

        # this smooth formulation works better if control starts from 0
        if U_init is None:
            self.action_sequence = torch.zeros_like(self.U)
        else:
            self.action_sequence = U_init
        self.U = torch.zeros_like(self.U)

    def get_params(self):
        return f"{super().get_params()} w={self.w_action_seq_cost} t={self.delta_t}"

    def shift_nominal_trajectory(self):
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        self.action_sequence = torch.roll(self.action_sequence, -1, dims=0)
        self.action_sequence[-1] = self.action_sequence[-2]  # add T-1 action to T

    def get_action_sequence(self):
        return self.action_sequence

    def reset(self):
        self.U = torch.zeros_like(self.U)
        self.action_sequence = torch.zeros_like(self.U)

    def change_horizon(self, horizon):
        if horizon < self.U.shape[0]:
            # truncate trajectory
            self.U = self.U[:horizon]
            self.action_sequence = self.action_sequence[:horizon]
        elif horizon > self.U.shape[0]:
            # extend with u_init
            extend_for = horizon - self.U.shape[0]
            self.U = torch.cat((self.U, self.u_init.repeat(extend_for, 1)))
            self.action_sequence = torch.cat((self.action_sequence, self.action_sequence[-1].repeat(extend_for, 1)))
        self.T = horizon

    def _bound_d_action(self, control):
        if self.u_max is not None:
            return torch.max(torch.min(control, self.u_max), self.u_min)  # action
        return control

    def _bound_action(self, action):
        if self.action_max is not None:
            return torch.max(torch.min(action, self.action_max), self.action_min)
        return action

    def _command(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()

        self._compute_weighting(cost_total)
        perturbations = torch.sum(self.omega.view(-1, 1, 1) * self.noise, dim=0)

        self.U = self.U + perturbations
        # U is now the lifted control space, so we integrate it
        self.action_sequence += self.U * self.delta_t

        action = self.get_action_sequence()[:self.u_per_command]
        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]
        return action

    def _compute_perturbed_action_and_noise(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        noise = self.noise_dist.rsample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_control = self.U + noise
        # naively bound control
        self.perturbed_control = self._bound_d_action(perturbed_control)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.perturbed_action = self.action_sequence + perturbed_control * self.delta_t
        self.perturbed_action = self._sample_specific_actions(self.perturbed_action)
        self.perturbed_action = self._bound_action(self.perturbed_action)

        self.noise = (self.perturbed_action - self.action_sequence) / self.delta_t - self.U

    def _compute_total_cost_batch(self):
        self._compute_perturbed_action_and_noise()
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv  # Like original paper

        # action difference as cost
        action_diff = self.u_scale * torch.diff(self.perturbed_action, dim=-2)
        action_smoothness_cost = torch.sum(torch.square(action_diff), dim=(1, 2))
        # handle non-homogeneous action sequence cost
        action_smoothness_cost *= self.w_action_seq_cost

        rollout_cost, self.states, actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions = actions / self.u_scale

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost + action_smoothness_cost
        return self.cost_total


class TimeKernel:
    """Kernel acting on the time dimension of trajectories for use in interpolation and smoothing"""

    def __call__(self, t, tk):
        raise NotImplementedError


class RBFKernel(TimeKernel):
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __repr__(self):
        return f"RBFKernel(sigma={self.sigma})"

    def __call__(self, t, tk):
        d = torch.sum((t[:, None] - tk) ** 2, dim=-1)
        k = torch.exp(-d / (1e-8 + 2 * self.sigma ** 2))
        return k


class KMPPI(MPPI):
    """MPPI with kernel interpolation of control points for smoothing"""

    def __init__(self, *args, num_support_pts=None, kernel: TimeKernel = RBFKernel(), **kwargs):
        super().__init__(*args, **kwargs)
        self.num_support_pts = num_support_pts or self.T // 2
        # control points to be sampled
        self.theta = torch.zeros((self.num_support_pts, self.nu), dtype=self.dtype, device=self.d)
        self.Tk = None
        self.Hs = None
        # interpolation kernel
        self.interpolation_kernel = kernel
        self.intp_krnl = None
        self.prepare_vmap_interpolation()

    def get_params(self):
        return f"{super().get_params()} num_support_pts={self.num_support_pts} kernel={self.interpolation_kernel}"

    def reset(self):
        super().reset()
        self.theta.zero_()

    def shift_nominal_trajectory(self):
        super().shift_nominal_trajectory()
        self.theta, _ = self.do_kernel_interpolation(self.Tk[0] + 1, self.Tk[0], self.theta)

    def do_kernel_interpolation(self, t, tk, c):
        K = self.interpolation_kernel(t.unsqueeze(-1), tk.unsqueeze(-1))
        Ktktk = self.interpolation_kernel(tk.unsqueeze(-1), tk.unsqueeze(-1))
        # print(K.shape, Ktktk.shape)
        # row normalize K
        # K = K / K.sum(dim=1).unsqueeze(1)

        # KK = K @ torch.inverse(Ktktk)
        KK = torch.linalg.solve(Ktktk, K, left=False)

        return torch.matmul(KK, c), K

    def prepare_vmap_interpolation(self):
        self.Tk = torch.linspace(0, self.T - 1, int(self.num_support_pts), device=self.d, dtype=self.dtype).unsqueeze(
            0).repeat(self.K, 1)
        self.Hs = torch.linspace(0, self.T - 1, int(self.T), device=self.d, dtype=self.dtype).unsqueeze(0).repeat(
            self.K, 1)
        self.intp_krnl = vmap(self.do_kernel_interpolation)

    def deparameterize_to_trajectory_single(self, theta):
        return self.do_kernel_interpolation(self.Hs[0], self.Tk[0], theta)

    def deparameterize_to_trajectory_batch(self, theta):
        assert theta.shape == (self.K, self.num_support_pts, self.nu)
        return self.intp_krnl(self.Hs, self.Tk, theta)

    def _compute_perturbed_action_and_noise(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        noise = self.noise_dist.rsample((self.K, self.num_support_pts))
        perturbed_control_pts = self.theta + noise
        # control points in the same space as control and should be bounded
        perturbed_control_pts = self._bound_action(perturbed_control_pts)
        self.noise_theta = perturbed_control_pts - self.theta
        perturbed_action, _ = self.deparameterize_to_trajectory_batch(perturbed_control_pts)
        perturbed_action = self._sample_specific_actions(perturbed_action)
        # naively bound control
        self.perturbed_action = self._bound_action(perturbed_action)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U

    def _command(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()

        self._compute_weighting(cost_total)
        perturbations = torch.sum(self.omega.view(-1, 1, 1) * self.noise_theta, dim=0)

        self.theta = self.theta + perturbations
        self.U, _ = self.deparameterize_to_trajectory_single(self.theta)

        action = self.get_action_sequence()[:self.u_per_command]
        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]
        return action


def run_mppi(mppi, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True):
    dataset = torch.zeros((retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d)
    total_reward = 0
    for i in range(iter):
        state = env.unwrapped.state.copy()
        command_start = time.perf_counter()
        action = mppi.command(state)            # 主接口
        elapsed = time.perf_counter() - command_start
        res = env.step(action.cpu().numpy())
        s, r = res[0], res[1]
        total_reward += r
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype)
        dataset[di, mppi.nx:] = action
    return total_reward, dataset
