import os
import gzip
import pickle
import numpy as np
import multiprocessing as mp
import copy
import torch

from invman.nn.utils import load_model
from invman.env.lost_sales import get_model_fitness


def create_model_id(model_name, args):
    model_detail = model_name.split("_")
    num_es_training = int(model_detail[-1])
    max_order_size = int(float(model_detail[-2]))
    training_horizon = int(float(model_detail[-5]))
    activation = model_detail[-6]
    id_ = (num_es_training, training_horizon, activation, max_order_size)

    args.activation = activation
    args.max_order_size = int(max_order_size)

    return id_, args


def eval_model(model_name, model, seeds, args):
    # evaluate the model over the provided sets of seeds
    args = copy.deepcopy(args)
    id_, results = create_model_id(model_name, args)
    eval_results = {}
    for seed in seeds:
        cost, _ = get_model_fitness(model, args, seed=seed)
        eval_results[seed] = -cost

    return {id_: eval_results, "model_name": model_name}


if __name__ == "__main__":
    from invman.config import get_config

    args = get_config()
    models_dir = args.trained_models_dir
    all_models_names = os.listdir(models_dir)

    models = {}
    for model_name in all_models_names:
        models[model_name] = load_model(os.path.join(models_dir, model_name))

    exp_seeds = np.random.randint(0, 10000, 100)
    args.horizon = int(1e6)
    exp_results = {}
    # Create a pool of workers to evaluate the models in parallel
    pool = mp.Pool(processes=args.mp_num_processors)

    results = [pool.apply_async(func=eval_model, args=(model_name, model, exp_seeds, args))
               for model_name, model in models.items()
               ]
    # collect the evaluation results of the models
    results = [res.get() for res in results]

    res_dir = os.path.join(args.out_dir, "Lost_sales_evaluation_results.pkl")
    with gzip.open(res_dir, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)



