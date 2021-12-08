
import numpy as np
import torch
from invman.env.lost_sales import LostSalesEnv


def get_model_fitness_lost_sales_problem(model, args, model_params=None, indiv_idx=-1, return_env=False,
                                         track_demand=False, seed=None, verbose=False):
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
