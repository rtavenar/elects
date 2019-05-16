import pandas as pd
import matplotlib.pyplot as plt

csvfile = "/home/marc/projects/early_rnn/data/early_reward.csv"

mori_accuracy_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/mori-accuracy-sr2-cf2.csv"
mori_earliness_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/mori-earliness-sr2-cf2.csv"

def main():

    mori = load_mori(mori_accuracy_path, mori_earliness_path)

    ours = pd.read_csv(csvfile)
    ours["score"] = ours["accuracy"] * (1 - ours["earliness"])

    mori_best_score = mori.iloc[mori.reset_index().groupby("Dataset")["score"].idxmax()]
    mori_best_score.index.name = "dataset"

    ours_best_score = ours.iloc[ours.reset_index().groupby("dataset")["score"].idxmax()]
    ours_best_score = ours_best_score.set_index("dataset")

    score = pd.merge(ours_best_score["score"], mori_best_score["score"], left_index=True, right_index=True)
    score.columns = ["ours", "mori"]

    filename = "/home/marc/projects/NeurIPS19/images/scatter/mori/scores.csv"
    print("writing "+filename)
    score.to_csv(filename)

    accuracy = pd.merge(ours_best_score["accuracy"], mori_best_score["accuracy"], left_index=True, right_index=True)
    accuracy.columns = ["ours", "mori"]

    filename = "/home/marc/projects/NeurIPS19/images/scatter/mori/accuracies.csv"
    print("writing "+filename)
    accuracy.to_csv(filename)

    earliness = pd.merge(ours_best_score["earliness"], mori_best_score["earliness"], left_index=True, right_index=True)
    earliness.columns = ["ours", "mori"]

    filename = "/home/marc/projects/NeurIPS19/images/scatter/mori/earliness.csv"
    print("writing "+filename)
    earliness.to_csv(filename)





    pass


def load_mori(mori_accuracy_path, mori_earliness_path):

    def load_single_alpha(col):

        mori_accuracy = pd.read_csv(mori_accuracy_path, sep=' ').set_index("Dataset")
        mori_earliness = pd.read_csv(mori_earliness_path, sep=' ').set_index("Dataset")/100
        mori = pd.concat([mori_accuracy[col], mori_earliness[col]], axis=1)

        mori.columns = ["accuracy", "earliness"]
        mori["score"] = mori["accuracy"] * (1 - mori["earliness"])

        return mori

    mori06 = load_single_alpha("a=0.6")
    mori07 = load_single_alpha("a=0.7")
    mori08 = load_single_alpha("a=0.8")
    mori09 = load_single_alpha("a=0.9")

    mori = pd.concat([mori06, mori07, mori08, mori09])

    return mori

if __name__=="__main__":
    main()