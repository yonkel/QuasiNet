from typing import Dict

from matplotlib import pyplot as plt


def export_to_latex(filename, item_list):
    with open(f"../results/{filename}", "w") as file:
        file.write(f"x y\n")
        for i, item in enumerate(item_list):
            file.write(f"{i} {item}\n")

def make_plt_graph(filename, losses: Dict[str, list]):
    for name, data in losses.items():
        plt.plot(data, label=name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.savefig(f"../results/{name}_{filename}")
        plt.show()

def make_Kaggle_file(item_list):
    with open(f"../results/titanic/outputs.csv", "w") as file:
        file.write("PassengerId,Survived\n")

        for i in range(len(item_list)):
            file.write(f"{i+892},{item_list[i]}\n")
