import pandas as pd
import ast


if __name__ == "__main__":
    batch = 4
    for file_name in [
        # f"2/parity2_results_2_3_4_5_6_7_8_batch{batch}.csv",
        # f"3/parity3_results_3_4_5_6_7_8_9_10_batch{batch}.csv",
        # f"4/parity4_results_4_5_6_7_8_9_10_11_12_batch{batch}.csv",
        # f"5/parity5_results_5_6_7_8_9_10_11_12_13_batch{batch}.csv",
        # f"6/parity6_results_6_7_8_9_10_11_12_13_14_batch{batch}.csv",
        # f"7/parity7_results_7_8_9_10_11_12_13_14_15_16_batch{batch}.csv",
    ]:

        df = pd.read_csv(file_name)
        df['epochs'] = df['epochs'].apply(ast.literal_eval)

        # ['parity_degree', 'hidden', 'converged', 'lr', 'batch_size', 'epochs']
        for i in range(df.shape[0]):
            row = df.iloc[i]

            # print(row["converged"], " & ", end=" ")

            filtered_numbers = [num for num in row['epochs'] if num != 1000]

            if len(filtered_numbers) == 0:
                print(1000, "ERROR")
            else:
                average = sum(filtered_numbers) / len(filtered_numbers)
                print(round(average, 2), end=" &  ")

        input()