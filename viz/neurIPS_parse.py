import pandas as pd
import matplotlib.pyplot as plt

csvfile = "/home/marc/projects/early_rnn/data/early_reward.csv"

mori_accuracy_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/mori-accuracy-sr2-cf2.csv"
mori_earliness_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/mori-earliness-sr2-cf2.csv"



def main():




    ours = pd.read_csv(csvfile)
    ours["score"] = ours["accuracy"] * (1 - ours["earliness"])

    ours_best_score = ours.iloc[ours.reset_index().groupby("dataset")["score"].idxmax()]
    ours_best_score = ours_best_score.set_index("dataset")

    filename="/home/marc/projects/early_rnn/data/ours_best_scores.csv"
    print("writing " + filename)
    ours_best_score.to_csv(filename)

    #### MORI
    mori = load_mori(mori_accuracy_path, mori_earliness_path)
    mori_best_score = mori.iloc[mori.reset_index().groupby("Dataset")["score"].idxmax()]
    mori_best_score.index.name = "dataset"

    score = pd.merge(ours_best_score["score"], mori_best_score["score"], left_index=True, right_index=True)
    score.columns = ["ours", "mori"]


    accuracy = pd.merge(ours_best_score["accuracy"], mori_best_score["accuracy"], left_index=True, right_index=True)
    accuracy.columns = ["ours", "mori"]


    earliness = pd.merge(ours_best_score["earliness"], mori_best_score["earliness"], left_index=True, right_index=True)
    earliness.columns = ["ours", "mori"]


    ### RELCLASS
    relclass = load_relclass()
    relclass_best_score = relclass.iloc[relclass.reset_index().groupby("Dataset")["score"].idxmax()]
    relclass_best_score.index.name = "dataset"

    score["relclass"] = relclass_best_score["score"]
    accuracy["relclass"] = relclass_best_score["accuracy"]
    earliness["relclass"] = relclass_best_score["earliness"]


    ### EDSC

    edsc = load_edsc()
    edsc_best_score = edsc.iloc[edsc.reset_index().groupby("Dataset")["score"].idxmax().dropna()]
    edsc_best_score.index.name = "dataset"

    score["edsc"] = edsc_best_score["score"]
    accuracy["edsc"] = edsc_best_score["accuracy"]
    earliness["edsc"] = edsc_best_score["earliness"]


    ### ECTS

    ects = load_ects()
    ects_best_score = ects.iloc[ects.reset_index().groupby("Dataset")["score"].idxmax()]
    ects_best_score.index.name = "dataset"

    score["ects"] = ects_best_score["score"]
    accuracy["ects"] = ects_best_score["accuracy"]
    earliness["ects"] = ects_best_score["earliness"]

    filename = "/home/marc/projects/NeurIPS19/images/scatter/sota/earliness.csv"
    print("writing "+filename)
    earliness.to_csv(filename)

    filename = "/home/marc/projects/NeurIPS19/images/scatter/sota/accuracies.csv"
    print("writing "+filename)
    accuracy.to_csv(filename)

    filename = "/home/marc/projects/NeurIPS19/images/scatter/sota/scores.csv"
    print("writing "+filename)
    score.to_csv(filename)

    print("metric & " + " & ".join(["mori","relclass","ects","edsc"]))
    print("score & " + " & ".join([print_comparison(score,approach) for approach in ["mori","relclass","ects","edsc"]]))
    print("accuracy & " + " & ".join([print_comparison(accuracy, approach) for approach in ["mori", "relclass", "ects", "edsc"]]))
    print("earliness & " + " & ".join([print_comparison(earliness, approach, greater_is_better=False) for approach in ["mori", "relclass", "ects", "edsc"]]))

def print_comparison(df,approach, greater_is_better=True):
    if greater_is_better:
        better=(df["ours"] > df[approach]).sum()
    else:
        better = (df["ours"] < df[approach]).sum()

    return str(better) + "/" + str(len(df[approach])-better)






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

def load_relclass():

    accuracy_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/relclass-accuracy-gaussian-naive-bayes.csv"
    earliness_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/relclass-earliness-gaussian-naive-bayes.csv"

    def load_single_alpha(col):

        accuracy = pd.read_csv(accuracy_path, sep=' ').set_index("Dataset")
        earliness = pd.read_csv(earliness_path, sep=' ').set_index("Dataset")/100
        relclass = pd.concat([accuracy[col], earliness[col]], axis=1)

        relclass.columns = ["accuracy", "earliness"]
        relclass["score"] = relclass["accuracy"] * (1 - relclass["earliness"])

        return relclass

    r001 = load_single_alpha("t=0.001")
    r01 = load_single_alpha("t=0.1")
    r05 = load_single_alpha("t=0.5")
    r09 = load_single_alpha("t=0.9")

    return pd.concat([r001, r01, r05, r09])

def load_edsc():

    accuracy_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/edsc-accuracy.csv"
    earliness_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/edsc-earliness.csv"

    def load_single_alpha(col):

        accuracy = pd.read_csv(accuracy_path, sep=' ').set_index("Dataset")
        earliness = pd.read_csv(earliness_path, sep=' ').set_index("Dataset")/100
        relclass = pd.concat([accuracy[col], earliness[col]], axis=1)

        relclass.columns = ["accuracy", "earliness"]
        relclass["score"] = relclass["accuracy"] * (1 - relclass["earliness"])

        return relclass

    return pd.concat([load_single_alpha("t=2.5"), load_single_alpha("t=3"), load_single_alpha("t=3.5")])

def load_ects():

    accuracy_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/ects-accuracy-strict-method.csv"
    earliness_path = "/home/marc/projects/early_rnn/data/UCR_Datasets/ects-earliness-strict-method.csv"

    def load_single_alpha(col):

        accuracy = pd.read_csv(accuracy_path, sep=' ').set_index("Dataset")
        earliness = pd.read_csv(earliness_path, sep=' ').set_index("Dataset")/100
        ects = pd.concat([accuracy[col], earliness[col]], axis=1)

        ects.columns = ["accuracy", "earliness"]
        ects["score"] = ects["accuracy"] * (1 - ects["earliness"])

        return ects

    return pd.concat([load_single_alpha(col) for col in ['sup=0', 'sup=0.05', 'sup=0.1', 'sup=0.2', 'sup=0.4', 'sup=0.8']])


if __name__=="__main__":
    main()