import numpy as np
import torch
from collections import deque
from scipy.stats import poisson


class LostSalesEnv():
    def __init__(self, demand_rate: float, lead_time: int = 2, max_order_size: int = 25, inventory_upper_bound: int = 200,
                 holding_cost: float = 1, shortage_cost: float = 4, horizon: int = int(1e5), procurement_cost=0,
                 track_demand=True, warm_up_periods_ratio=0.2):
        self.demand_rate = demand_rate
        self.demand_dist_name = "Poisson"
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.procurement_cost = procurement_cost
        self.action_space_dim = self.max_order_size = max_order_size
        self.inventory_upper_bound = inventory_upper_bound
        self.state_space_dim = self.lead_time = lead_time
        self.lead_time_orders = deque(maxlen=lead_time)
        self.current_epoch = 0
        self.done = False
        self.warm_up_periods = int(warm_up_periods_ratio * horizon)
        self.gamma = 0.995
        self.horizon = horizon
        self.track_demand = track_demand
        self.demand_probs, self.demand_lb, self.demand_ub = self.get_demand_prob_vector()
        self.reset()

    def initialize_env(self):
        """
        Initialize the environment
        """
        self.current_inventory_level = 2 * self.demand_rate
        for _ in range(self.lead_time):
            # order randomly uniformly randomly between [1, max_order_size]
            order_quantity = np.random.randint(low=1, high=self.max_order_size)
            self.lead_time_orders.append(order_quantity)
            epoch_demand = np.random.poisson(self.demand_rate)
            self.current_inventory_level  = np.max((0, self.current_inventory_level - epoch_demand))

    def reset(self,):
        self.total_cost = 0
        self.epoch_costs = []
        self.y_d_data = {}
        self.q_L_data = {}
        self.m2_q_L_data = {}
        if self.track_demand:
            self.horizon_demand = np.random.poisson(lam=self.demand_rate, size=self.horizon)
        else:
            self.horizon_demand = None
        self.initialize_env()

        return self.norm_state

    def is_valid_action(self, action):
        if action > self.max_order_size:
            return False
        return True

    def get_realized_demand(self):
        if self.horizon_demand is not None:
            return self.horizon_demand[self.current_epoch]
        else:
            return np.random.poisson(self.demand_rate)

    def get_state_dim(self):
        return self.lead_time

    def is_done(self):
        return self.done

    def get_epoch_cost(self, epoch_demand):
        #print("E", env.current_epoch, "demand", epoch_demand, "inventory", env.current_inventory_level,
        #      "orders", env.lead_time_orders)
        if epoch_demand < self.current_inventory_level:
            self.current_inventory_level -= epoch_demand
            epoch_cost = self.current_inventory_level * self.holding_cost
        else:
            lost_sales = epoch_demand - self.current_inventory_level
            epoch_cost = self.shortage_cost * lost_sales  # get the lost sales cost
            self.current_inventory_level = 0  # set the current inventory level to zero

        return epoch_cost

    def update_lead_time_orders(self, order_quantity):
        self.arriving_order = self.lead_time_orders.popleft()
        self.lead_time_orders.append(order_quantity)

        return self.arriving_order

    @property
    def state(self):
        # State needs to be used always before calling the step to make sure we are using the inventory level of the
        # previous period
        state = list(self.lead_time_orders)
        state[0] += self.current_inventory_level
        return state

    @property
    def norm_state(self):
        return np.array(self.state)/self.max_order_size

    def step(self, order_quantity):
        arriving_orders = self.update_lead_time_orders(order_quantity)  # order arrives and lead_time orders gets updated
        self.current_inventory_level += arriving_orders  # inventory gets updated
        epoch_demand = self.get_realized_demand()

        epoch_cost = self.get_epoch_cost(epoch_demand)

        self.epoch_costs.append(epoch_cost)

        self.current_epoch += 1
        if self.current_epoch >= self.horizon:
            self.done = True

        return self.state, epoch_cost, self.done
        #return self.norm_state, epoch_cost, self.done

    def get_one_hot_encoded_state(self, state):
        d = self.inventory_upper_bound + self.max_order_size
        s = np.zeros((1, d))
        s[0, state[0]] = 1
        s[0, self.inventory_upper_bound + state[1]] = 1

        return s

    @property
    def avg_total_cost(self, after_warmup=True,):
        if after_warmup:
            return np.mean(self.epoch_costs[self.warm_up_periods:])
        else:
            return np.mean(self.epoch_costs)

    def get_demand_lower_upper_bound(self, eps=1e-10):

        return poisson.interval(alpha=1 - eps, mu=self.demand_rate)

    def get_demand_prob_vector(self):
        lb, ub = self.get_demand_lower_upper_bound()
        demand_range = np.arange(lb, ub)
        demand_probs = poisson.pmf(demand_range, self.demand_rate)

        return demand_probs, int(lb), int(ub)

    def get_cumulative_demand_l_L(self, k, l=0):
        if self.demand_dist_name == "Poisson":
            # Sum of indepedent Poisson distributions is equal to a Poisson distribution with the summation of their rates
            rate = (self.lead_time - l + 1) * self.demand_rate
            return poisson.cdf(k=k, mu=rate)

    def get_critical_fractile(self):
        critical_fractile = (self.procurement_cost + self.holding_cost) / (self.holding_cost + self.shortage_cost)

        return critical_fractile

    @property
    def critical_fractile(self):
        return self.get_critical_fractile()

    def get_order_pipeline_partial_sum(self, state, l):
        if l == self.lead_time:
            return 0
        return sum(list(state)[l:])


def get_model_fitness(model, args, model_params=None, indiv_idx=-1, return_env=False, track_demand=False, seed=None,
                      verbose=False):
    if model_params is not None:
        model.set_model_params(model_params)

    if seed is None:
        if hasattr(args, "seed"):
            if verbose:
                print(f"The seed from the args is set to {args.seed}")
            np.random.seed(args.seed)
        else:
            raise NotImplementedError
    else:
        if verbose:
            print(f"The seed is set to {seed}")
        np.random.seed(seed)

    env = LostSalesEnv(demand_rate=args.demand_rate, lead_time=args.lead_time, horizon=args.horizon,
                       max_order_size=args.max_order_size, holding_cost=args.holding_cost,
                       shortage_cost=args.shortage_cost, track_demand=track_demand)
    state = env.norm_state
    done = False
    while not done:
        state = torch.FloatTensor(state)
        order_quantity = model(state)
        state, epoch_cost, done = env.step(order_quantity=order_quantity)
    if return_env:
        return -env.avg_total_cost, env
    return -env.avg_total_cost, indiv_idx


if __name__ == "__main__":
    from invman.config import get_config
    from invman.heuristics.lost_sales_heuristics import get_heuristic_policy_cost
    args = get_config()

    env = LostSalesEnv(demand_rate=5., lead_time=2, horizon=int(1e6))
    input_dim = env.state_space_dim
    output_dim = env.action_space_dim
    from invman.nn.policy_net import PolicyNet

    model = PolicyNet(input_dim=2, hidden_dim=[32, 16], output_dim=20)
    """
    done = False
    state_action = {}
    while not done:
        if tuple(env.state) in state_action:
            order_quantity = state_action[tuple(env.state)]
        else:
            order_quantity = env.get_myopic_1_order_quantity()
            state_action[tuple(env.state)] = order_quantity
        state, epoch_cost, done = env.step(order_quantity=order_quantity)
    """
    args.horizon = int(1e6)
    args.seed = 123

    env_svbs, svbs_tc, state_action_svbs = get_heuristic_policy_cost(args, heuristic="standard_vector_base_stock")
    print(svbs_tc)
    env_m1, m1_tc, state_action_m1 = get_heuristic_policy_cost(args, heuristic="myopic1")
    print(m1_tc)
    env_m2, m2_tc, state_action_m2 = get_heuristic_policy_cost(args, heuristic="myopic2")
    print(m2_tc)




