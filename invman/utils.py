
import os
import pickle
import gzip


def save_init_args(init_method):
    def wrapper(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
        init_method(self, *args, **kwargs)
    return wrapper


def write_log(log, model, args):
    log_file_name = f"log_{args.problem}_{model._get_name()}_{model.num_params}_{args.desc}.txt"
    os.makedirs(args.log_dir, exist_ok=True)
    log_file_dir = os.path.join(args.log_dir, log_file_name)
    with open(log_file_dir, "a") as f:
        f.write(log)
        f.write("\n")


def save_es_reward_hist(es, fitness_hist, args):
    os.makedirs(args.log_dir, exist_ok=True)
    es_model_name = args.desc + "cma"
    es_model_dir = os.path.join(args.log_dir, es_model_name)
    with gzip.open(es_model_dir, "wb") as f:
        pickle.dump(es, f)

    es_reward_hist_name = args.desc+ "cma_reward_hist"
    es_reward_hist_dir = os.path.join(args.log_dir, es_reward_hist_name)
    with gzip.open(es_reward_hist_dir, "wb") as f:
        pickle.dump(fitness_hist, f)


def save_model_solutions(model, solutions, episode, args, save_solutions=False, es_parameter_type='avg'):
    model_dir_name = f'{args.problem}_{model._get_name()}_{model.num_params}_{args.policy_network_size}_{es_parameter_type}_{args.desc}_{episode}'
    os.makedirs(args.trained_models_dir, exist_ok=True)
    model_save_dir = os.path.join(args.trained_models_dir, model_dir_name)
    model.save(model_save_dir, override=True)
    if save_solutions:
        solutions_name = f"{args.problem}_solutions_{model._get_name()}_{model.num_params}_{args.desc}"
        solutions_save_dir = os.path.join(args.trained_models_dir, model_dir_name, solutions_name)
        with gzip.open(solutions_save_dir, "wb") as f:
            pickle.dump(solutions, f, protocol=pickle.HIGHEST_PROTOCOL)