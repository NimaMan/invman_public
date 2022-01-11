import numpy as np


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class CMAES:
    """CMA-ES wrapper"""
    def __init__(
        self,
        num_params,  # number of model parameters
        sigma_init=0.10,  # initial standard deviation
        popsize=255,  # population size
        weight_decay=0.00,
        param_scales = None
    ):  # weight decay coefficient

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None
        self.param_scales = np.array([1]) if param_scales is None else param_scales

        import cma

        self.es = cma.CMAEvolutionStrategy(
            #self.num_params * [0],
            np.random.randn(self.num_params),
            self.sigma_init,
            {
                "popsize": self.popsize,
            },
        )

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.sqrt(np.mean(sigma * sigma))

    def ask(self):
        """returns a list of parameters"""
        self.solutions = np.array(self.es.ask())
        self.solutions *= self.param_scales[None,:]
        return self.solutions

    def tell(self, reward_table_result):
        reward_table = -np.array(reward_table_result)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay
        self.es.tell(
            self.solutions/self.param_scales[None,:], (reward_table).tolist()
        )  # convert minimizer to maximizer.

    def current_param(self):
        return self.es.result[5] * self.param_scales  # mean solution, presumably better with noise

    def best_param(self):
        return self.es.result[0] * self.param_scales  # best evaluated solution

    def result(
        self,
    ):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])


