from invman.nn.policy_net import PolicyNet
from invman.env.lost_sales import get_model_fitness
from invman.heuristics.lost_sales_heuristics import get_heuristic_policy_cost
from invman.es_mp import train


def run(args):
    model = PolicyNet(input_dim=args.lead_time, hidden_dim=[20, 10], output_dim=args.max_order_size,
                      activation=args.activation)
    print("Training started", args.desc)
    model, fitness_hist = train(model=model, get_model_fitness=get_model_fitness, args=args)


if __name__ == "__main__":
    from invman.config import get_config
    args = get_config()
    env_svbs, svbs_tc, state_action_svbs = get_heuristic_policy_cost(args, heuristic="standard_vector_base_stock",
                                                                     seed=1234)
    max_order_size = max(state_action_svbs.values()) + 1
    desc = args.desc
    args.sigma_init = 5
    for env_horizon in [1e2, 5e2, 1e3, 5e3, 1e4]:
        args.horizon = int(env_horizon)
        for activation in ["gelu", 'selu']:
            args.activation = activation
            for order_ub in [max_order_size, 2*max_order_size, 3*max_order_size]:
                args.max_order_size = int(order_ub)
                args.desc = desc + f"_lead_time_{args.lead_time}_{activation}" + f'_{env_horizon}' + f'_order_ub_{order_ub}'
                args.exp_des = args.desc
                run(args)
                print(f"finished {args.desc}")