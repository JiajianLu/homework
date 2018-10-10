import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

"""

python plot.py data/lb_no_rtg_dna_CartPole-v0 data/lb_rtg_dna_CartPole-v0 data/lb_rtg_na_CartPole-v0 --legend trajectory_reward reward_to_go reward_to_go_center
--value AverageReturn
python plot.py data/sb_no_rtg_dna_CartPole-v0 data/sb_rtg_dna_CartPole-v0 data/sb_rtg_na_CartPole-v0 --legend trajectory_reward reward_to_go reward_to_go_center

python plot.py data/hc_b_r_10000_02_HalfCheetah-v2 data/hc_b_r_10000_01_HalfCheetah-v2 data/hc_b_r_10000_005_HalfCheetah-v2 data/hc_b_r_30000_02_HalfCheetah-v2 data/hc_b_r_30000_01_HalfCheetah-v2 data/hc_b_r_30000_005_HalfCheetah-v2 data/hc_b_r_50000_02_HalfCheetah-v2 data/hc_b_r_50000_01_HalfCheetah-v2 data/hc_b_r_50000_005_HalfCheetah-v2

python plot.py data/hc_b_r_095_50000_01_HalfCheetah-v2 data/hc_b_r_095_50000_01_rtg_HalfCheetah-v2 data/hc_b_r_095_50000_01_nn_HalfCheetah-v2 data/hc_b_r_095_50000_01_rtg_nn_HalfCheetah-v2

--value AverageReturn 
--legend Learning_Curve 
Using the plotter:  

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/hc_b_r_InvertedPendulum-v2_20-09-2018_02-17-33 --value AverageReturn
    python plot.py data/ll_b40000_r0.005_LunarLanderContinuous-v2 --value AverageReturn --legend Learning_Curve

python plot.py data/hc_b_r_HalfCheetah-v2_1_05 data/hc_b_r_HalfCheetah-v2_19-09-2018_18-13-34 

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.

python plot.py data/sb_no_rtg_dna_CartPole-v0_19-09-2018_11-37-54 data/sb_no_rtg_dna_CartPole-v0_19-09-2018_12-21-19

Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/sb_no_rtg_dna_CartPole-v0 data/sb_rtg_dna_CartPole-v0 data/sb_rtg_na_CartPole-v0

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""

def plot_data(data, value="AverageReturn"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition")
    plt.legend(loc='best').draggable()
    plt.show()


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root,'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']
            
            log_path = os.path.join(root,'log.txt')
            experiment_data = pd.read_table(log_path)

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )        
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
                )

            datasets.append(experiment_data)
            unit += 1

    return datasets
#/data/

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            data += get_datasets(logdir)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    for value in values:
        plot_data(data, value=value)

if __name__ == "__main__":
    main()
