import os
import argparse
from dotenv import load_dotenv


def get_config():

    load_dotenv()
    here = os.path.abspath(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='Lost sales inventory management')

    parser.add_argument("--problem", default="LS", type=str, help="Problem name")

    # Opt
    parser.add_argument("--training_method", default="cma", type=str, help="training method")
    parser.add_argument("--training_episodes", default=1000, type=int, help="training method")
    parser.add_argument("--mp_num_processors", default=os.getenv("MP_NUM_PROCESSORS"), type=int, help="training method")
    parser.add_argument("--sigma_init", default=5, type=float, help="initial sigma of cma-es")
    parser.add_argument("--es_population", default=50, type=int, help="Number of es population")

    # Env
    parser.add_argument("--demand_rate", default=5, type=float, help="demand rate")
    parser.add_argument("--max_order_size", default=25, type=int, help="maximum order size")
    parser.add_argument("--lead_time", default=2, type=int, help="lead time")
    parser.add_argument("--shortage_cost", default=4, type=float, help="shortage cost of the system")
    parser.add_argument("--holding_cost", default=1, type=float, help="holding cost of the system")
    parser.add_argument("--horizon", default=int(5e2), type=int, help="number of simulation epochs")

    # Policy Net
    parser.add_argument("--hidden_dim", default=[20, 10], type=list, help="number of neuron in each layer of the neural network")

    # Directories
    parser.add_argument("--out_dir", type=str, default=os.getenv("OUTDIR"),
                        help="the output directory for generating data", )
    parser.add_argument("--log_dir", default=os.getenv("LOGDIR"), help="Directory to write TensorBoard information to", )
    parser.add_argument("--trained_models_dir", default=os.getenv("MODELDIR"),
                        help="Directory to write TensorBoard information to", )

    # Decription
    parser.add_argument("--desc", default="lost_sales_inv_man", type=str, help="experiment description")

    # Misc
    parser.add_argument("--model_save_step", default=1000, type=int, help="saves the model every model_save_step ")

    # return args
    return parser.parse_args()
