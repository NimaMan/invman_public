
# Inventory Management with Reinforcement Learning and Evolution Strategies

## Installation 
In order to install the library:
- open Anaconda Prompt
- go to the directory where the `setup.py` file is located using `cd [your directory]`
- run `pip install -e .`

## Setting the Log and Model Directory and number of parallel processors
Before running the training you have to set the directories for saving the models and the log 
## Train a Neural network to manage inventory in the lost sales problem
You can set the parameters of the lost sales problem in two ways. 

If you are running the training inside an IDE such as 
Spyder or Pycharm you may use the config file located in `invman/config.py` to set the parameters of the problem.

If you are running the training from the command line  you may set the parameters directly from there. 
For instance, if you would like to  change the lead time of the problem you can run `python train_ls_es.py --lead_time 3`. 
For this, you first need to go to the `scripts` directory where the training file is located.
A full list of current supported parameters can be seen below:
```python
parser.add_argument("--training_method", default="cma", type=str, help="training method")
parser.add_argument("--training_episodes", default=1000, type=int, help="training method")
parser.add_argument("--mp_num_processors", default=os.getenv("MP_NUM_PROCESSORS"), type=int, help="training method")
parser.add_argument("--sigma_init", default=1, type=float, help="initial sigma of cma-es")
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
```

In order to train 




