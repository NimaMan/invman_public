

import os
import pickle 
import gzip
import numpy as np
import matplotlib.pyplot as plt 




if __name__ == "__main__":
    from invman.config import get_config
    args = get_config()
    exp_dir = r"C:\Users\20204069\OneDrive - Koc Universitesi\ML\inventory_management\exp_results"
    res_dir = os.path.join(exp_dir, "Lost_sales_evaluation_results.pkl")
    
    with gzip.open(res_dir, "rb") as f:
        results = pickle.load(f)
        
    es_training_episodes = [1000, 2000, 3000, 4000, 5000]
    env_horizon = [100, 500, 1000, 5000]
    output_actions = [8, 16, 24]
    e = {}
    for res in results:
        for key in res.keys():
            if len(key) == 4:
                    if key[2] == "gelu":
                        e[key] = list(res[key].values())
            
    
    import pandas as pd
    df = pd.DataFrame(e)

    stats = df.describe().transpose().reset_index()
    stats.columns = ['es_training_episodes', 'env_horizon', 'activation', 
                     'output_space', 'count', 'mean', 'std','min', '25%', 
                     '50%', '75%', 'max']
    
    #stats.drop(index=67, inplace=(True))
    #s = stats[stats["output_space"] == 24]
    s = stats[stats["activation"] == "relu"]
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
    for out, ax in zip(output_actions, axes):
        ls = []
        labels = []                 
        for key in e.keys():
            if (key[-1] == out) and (key[0] in [1000, 5000]):
                    ls.append(e[key])
                    labels.append(key[:2])    
        
        sorted_ls = sorted(zip(labels, ls))
        labels = [x for x, _ in sorted_ls]
        ls = [y for _, y in sorted_ls]
        
        #ax.plot(e[key], kind="bar")
        #ax.boxplot(ls, showfliers=False, labels=labels, )
        ax.violinplot(ls, showmeans=True)
        #ax.errorbar(x=labels, y=ls, yerr=np.std(ls,  axis=1))
        ax.set_title(f"Size of Action Space $(d_o)$ = {out}")
        ax.yaxis.grid(True)
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels)
        
        ax.hlines(y=4.73, xmin=.5, xmax=len(ls)+.5, color="r", linestyles='--', label="Optimal")
        ax.hlines(y=4.82, xmin=.5, xmax=len(ls)+.5, color="g", linestyles='--', label="Myopic2")
        ax.legend()
        ax.set_ylabel('Average Cost')
    ax.set_xlabel('(Number of Traning Epochs, Environment Horizon)')        
    plt.tight_layout()
    fig_dir = os.path.join(exp_dir, "figures", "Lost_sales_p_4_l_4.jpg")
    fig.savefig(fig_dir)
                        
                                        
    