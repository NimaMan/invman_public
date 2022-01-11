import numpy as np
import torch
import multiprocessing as mp
import time
from invman.es import CMAES
from invman.utils import write_log, save_model_solutions, save_es_reward_hist


def get_es_optimizer(model, args):
    if args.training_method == "cma":
        es = CMAES(
            model.num_params,
            sigma_init=args.sigma_init,
            popsize=args.es_population,
        )
    else:
        raise NotImplementedError

    return es


def train(model, get_model_fitness, args):
    """
    :param model: The neural network model
    :param get_model_fitness: A method that outputs the reward associated with a set of flat parameters  of the model
            coming from the es_algos algorithm
    :param args: Other related parameters
    :return: None
    """
    episodes = args.training_episodes
    #model = model.to(args.device)
    history = []
    fitness_hist = []

    es = get_es_optimizer(model, args)
    print(f"Starting {args.training_method} with {model.num_params} parameters ")
    start = time.time()
    pool = mp.Pool(processes=args.mp_num_processors)
    for episode in range(1, episodes+1):
        solutions = es.ask()
        args.seed = np.random.randint(1, 100000)  # generates the same random keys for all es population

        results = [
            pool.apply_async(
                get_model_fitness,
                args=(model, args, solution, indiv_id),
            )
            for indiv_id, solution in enumerate(solutions)
        ]

        pop_fitness = [result.get() for result in results]  # Get process results from the output queue
        pop_fitness = sorted(pop_fitness, key=lambda x: x[1])
        es_fitness = np.array([f for f, idx in pop_fitness])

        # UPDATE the parameters
        es.tell(es_fitness)
        result = es.result()
        # Record the results of the episode
        fitness_hist.append(es_fitness)
        h = (
            f"e{episode} reward -> best: {np.max(es_fitness):.3f} mean: {np.mean(es_fitness):.3f}, "
            f"std: {np.std(es_fitness):.3f}"
        )
        print(h)
        history.append(h)
        write_log(h, model, args)
        if episode % args.model_save_step == 0:
            curr_solution = es.current_param()
            model = model.set_model_params(curr_solution)
            save_model_solutions(model, solutions, episode, args, save_solutions=False)

    curr_solution = es.current_param()
    model = model.set_model_params(curr_solution)
    save_model_solutions(model, solutions, 'final', args, save_solutions=True, es_parameter_type='Avg')
    print(f"the optimization ended in {(time.time() - start)/60}")
    save_es_reward_hist(fitness_hist=fitness_hist, es=es, args=args)

    return model, fitness_hist
