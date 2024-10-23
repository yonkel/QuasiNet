import pandas as pd

from experiments.parity import test_parity


if __name__ == '__main__':
    p = 2
    h_list = [2, 3, 4, 5]

    # TODO add hyperparameters to test_parity so the customization can be done on this level
    results = test_parity(parity_degree=p, hidden_layers=h_list)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/parity{p}_results_{'_'.join([str(h) for h in h_list])}.csv")