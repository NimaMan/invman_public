"""
Source: Old and New Methods for Lost-Sales Inventory Systems, Paul Zipkin
Implementation of the Myopic1, Myopic2, and Standard Vector Base Stock Policies
"""
import numpy as np
from invman.env.lost_sales import LostSalesEnv


class LostSalesHeuristicPolicies():

    def __init__(self, env: LostSalesEnv):
        self.env = env

        self.y_d_data = {}
        self.q_L_data = {}
        self.m2_q_L_data = {}

    def get_x_plus_l_1(self, x, demand, l):
        x_plus = x[:-1]
        x_plus[0] = np.max((0, x[0] - demand)) + x[1]
        for i in range(1, l):
            x_plus[i] = x[i + 1]

        return x_plus

    def get_one_period_cost(self, y):
        procurment_cost = self.env.procurement_cost * y
        d = np.arange(self.env.demand_lb, self.env.demand_ub)
        y_d_plus = y - d
        y_d_minus = y - d
        y_d_plus[y_d_plus < 0] = 0
        y_d_minus[y_d_minus > 0] = 0
        expected_overage = np.dot(self.env.demand_probs, y_d_plus)
        overage_cost = self.env.holding_cost * expected_overage

        expected_underage = np.dot(self.env.demand_probs, y_d_minus)
        underage_cost = self.env.shortage_cost * expected_underage

        return procurment_cost + abs(overage_cost) + abs(underage_cost)

    def get_q_l(self, x, l):
        if l == 0:
            if x[0] in self.y_d_data:
                q_0_x_0_plus = self.y_d_data[x[0]]
            else:
                q_0_x_0_plus = self.get_one_period_cost(x[0])
                self.y_d_data[x[0]] = q_0_x_0_plus
            return q_0_x_0_plus
        else:
            if (l, *x) in self.y_d_data:
                q_l_x_l = self.y_d_data[(l, *x)]
            else:
                q_l_x_l = 0
                for demand in range(self.env.demand_lb, self.env.demand_ub):
                    x_plus_l_1 = self.get_x_plus_l_1(x=x, demand=demand, l=l)
                    q_l_x_l_1_plus = self.get_q_l(x=x_plus_l_1, l=l - 1)
                    q_l_x_l += q_l_x_l_1_plus * self.env.demand_probs[demand - self.env.demand_lb]

                self.y_d_data[(l, *x)] = q_l_x_l
            return self.env.gamma * q_l_x_l

    def get_Q_L_x_L_from_state(self, state, order_quantity):
        x_L = state + [order_quantity]
        if (order_quantity, *state) in self.q_L_data:
            Q_L_x_L = self.q_L_data[(order_quantity, *state)]
        else:
            Q_L_x_L = 0
            for demand in range(self.env.demand_lb, self.env.demand_ub):
                x_plus_l_1 = self.get_x_plus_l_1(x=x_L, demand=demand, l=self.env.lead_time)
                q_l_x_l_plus = self.get_q_l(x=x_plus_l_1, l=self.env.lead_time - 1)
                Q_L_x_L += q_l_x_l_plus * self.env.demand_probs[demand - self.env.demand_lb]

            self.q_L_data[(order_quantity, *state)] = Q_L_x_L

        Q_L_x_L = self.env.gamma * Q_L_x_L
        return Q_L_x_L

    def get_Q_L_x_L(self, order_quantity):
        state = self.env.state
        return self.get_Q_L_x_L_from_state(state, order_quantity)

    def get_myopic_1_order_quantity(self, state, return_qhat=False):
        order_quantity = 0
        c1 = self.get_Q_L_x_L_from_state(state, order_quantity)
        c2 = np.inf
        while c2 > c1:
            order_quantity += 1
            c2 = c1
            c1 = self.get_Q_L_x_L_from_state(state, order_quantity)
        if return_qhat:
            return order_quantity-1, c2
        else:
            return order_quantity-1

    def get_myopic_2_q_L_x_L(self, state, order_quantity):
        x_L = state + [order_quantity]
        if (order_quantity, *state) in self.m2_q_L_data:
            q_hat_z = self.m2_q_L_data[(order_quantity, *state)]
        else:
            q_hat_z = self.get_Q_L_x_L_from_state(state, order_quantity)
            Q_L_x_L = 0
            for demand in range(self.env.demand_lb, self.env.demand_ub):
                x_plus = self.get_x_plus_l_1(x=x_L, demand=demand, l=self.env.lead_time)
                z, q_hat = self.get_myopic_1_order_quantity(x_plus, return_qhat=True)
                Q_L_x_L += q_hat * self.env.demand_probs[demand - self.env.demand_lb]

            q_hat_z = self.env.gamma * Q_L_x_L + q_hat_z
            self.m2_q_L_data[(order_quantity, *state)] = q_hat_z

        return q_hat_z

    def get_myopic_2_order_quantity(self, state, return_qhat=False):
        order_quantity = 0
        c1 = self.get_myopic_2_q_L_x_L(state, order_quantity)
        c2 = np.inf
        while c2 > c1:
            order_quantity += 1
            c2 = c1
            c1 = self.get_myopic_2_q_L_x_L(state, order_quantity)
        if return_qhat:
            return order_quantity-1, c2
        else:
            return order_quantity-1

    def get_order_pipeline_partial_sum(self, l, state):
        if l == self.env.lead_time:
            return 0
        return sum(list(state)[l:])

    def get_standard_vector_base_Stock_policy(self):
        sbar = np.zeros(self.env.lead_time + 1)
        for l in range(self.env.lead_time + 1):
            s = 0
            while (1 - self.env.get_cumulative_demand_l_L(k=s, l=l)) >= self.env.critical_fractile:
                s += 1
            sbar[l] = s

        return sbar

    def get_standard_vector_base_stock_policy_order_quantity(self, state):
        z_x = np.zeros(self.env.lead_time + 1)
        sbar = self.get_standard_vector_base_Stock_policy()

        for l in range(self.env.lead_time + 1):
            v_l = self.get_order_pipeline_partial_sum(l=l, state=state)
            z_x[l] = sbar[l] - v_l

        return np.max((0, np.min(z_x)))


def get_heuristic_policy_cost(args, env=None, heuristic="myopic1", seed=1234):
    if hasattr(args, "seed"):
        np.random.seed(args.seed)
    else:
        np.random.seed(seed)
    if env is None:
        env = LostSalesEnv(demand_rate=args.demand_rate, lead_time=args.lead_time, horizon=args.horizon,
                           max_order_size=args.max_order_size, holding_cost=args.holding_cost,
                           shortage_cost=args.shortage_cost)
    elif not isinstance(env, LostSalesEnv):
        raise NotImplementedError

    policy = LostSalesHeuristicPolicies(env=env)
    done = False
    state_action = {}
    while not done:
        if tuple(env.state) in state_action:
            order_quantity = state_action[tuple(env.state)]
        else:
            if heuristic == "myopic1":
                order_quantity = policy.get_myopic_1_order_quantity(env.state)
            elif heuristic == "myopic2":
                order_quantity = policy.get_myopic_2_order_quantity(env.state)
            elif heuristic == "standard_vector_base_stock":
                order_quantity = policy.get_standard_vector_base_stock_policy_order_quantity(env.state)
            else:
                raise NotImplementedError

            state_action[tuple(env.state)] = order_quantity

        state, epoch_cost, done = env.step(order_quantity=order_quantity)

    return env, -env.avg_total_cost, state_action


if __name__ == "__main__":
    from invman.config import get_config

    args = get_config()
    args.seed = 1234
    args.horizon = int(1e6)
    env_2, m2_tc, state_action2 = get_heuristic_policy_cost(args, heuristic="myopic2")
    print(m2_tc)
    env_1, m1_tc, state_action_1 = get_heuristic_policy_cost(args, heuristic="myopic1")
    print(m1_tc)
    env_svbs, svbs_tc, state_action_svbs = get_heuristic_policy_cost(args, heuristic="standard_vector_base_stock")
    print(m2_tc, svbs_tc)
    #env1, m1_tc, state_action1 = get_heuristic_policy_average_cost(args, heuristic="myopic1")
    #env_svbs, svbs_tc, state_action_svbs = get_heuristic_policy_average_cost(args, heuristic="standard_vector_base_Stock")
    #print(m1_tc, m2_tc, svbs_tc)
    # (4, 6, 4, 6) -> 4


