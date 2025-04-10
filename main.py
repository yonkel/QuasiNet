import pandas as pd

from experiments.parity import test_parity

if __name__ == '__main__':
    p = 8
    h_list = [13,14,15,16,17]
    batch = 4

    # TODO add hyperparameters to test_parity so the customization can be done on this level
    results = test_parity(parity_degree=p,
                          hidden_layers=h_list,
                          repeats=0,
                          learning_rate=0.5,
                          batch_size=batch,
                          zero_label=-1,
                          max_epochs=1000,
                          )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/parity{p}_results_{'_'.join([str(h) for h in h_list])}_batch{batch}.csv", index=False)
