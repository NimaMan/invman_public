from invman.nn.policy_net import PolicyNet
from invman.env.lost_sales import get_model_fitness
from invman.heuristics.lost_sales_heuristics import get_heuristic_policy_cost
from invman.es_mp import train


if __name__ == "__main__":
    from invman.config import get_config

    args = get_config()
    args.exp_des = args.desc
    env_svbs, svbs_tc, state_action_svbs = get_heuristic_policy_cost(args, heuristic="standard_vector_base_stock")
    max_order_size = max(state_action_svbs.values()) + 1
    args.max_order_size = 25
    model = PolicyNet(input_dim=args.lead_time, hidden_dim=[10, 20], output_dim=args.max_order_size)
    args.sigma_init = 5
    print("Training started", max_order_size)
    model, fitness_hist = train(model=model, get_model_fitness=get_model_fitness, args=args)

    args.horizon = int(1e5)
    args.seed = 42
    c, env = get_model_fitness(model, args, return_env=True)
    print("Evaluation Cost", c)

