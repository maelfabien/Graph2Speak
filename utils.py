import networkx as nx
import nx_altair as nxa
import numpy as np
import itertools
from pyvis.network import Network
import pandas as pd


def ep_dicts(season):
    """
    Store the list of speakers and their corresponding ID from Kaldi
    """

    if season == "s01e07":
        dict_spk = {
            "eddiewillows": "1001_csi",
            "jesseoverton": "1002_csi",
            "conradecklie": "1003_csi",
            "sheriff_brianmobley": "1004_csi",
            "tedgoggle": "1005_csi",
            "lie_detector_operator": "1006_csi",
            "nick": "1007_csi",
            "warrick": "1008_csi",
            "det_oriley": "1009_csi",
            "brass": "1010_csi",
            "tinacollins": "1011_csi",
            "sara": "1012_csi",
            "catherine": "1013_csi",
            "grissom": "1014_csi",
        }
        spk_dict = {
            "1001_csi": "eddiewillows",
            "1002_csi": "jesseoverton",
            "1003_csi": "conradecklie",
            "1004_csi": "sheriff_brianmobley",
            "1005_csi": "tedgoggle",
            "1006_csi": "lie_detector_operator",
            "1007_csi": "nick",
            "1008_csi": "warrick",
            "1009_csi": "det_oriley",
            "1010_csi": "brass",
            "1011_csi": "tinacollins",
            "1012_csi": "sara",
            "1013_csi": "catherine",
            "1014_csi": "grissom",
        }

    elif season == "s01e08":
        dict_spk = {
            "paigeharmon": "1001_csi",
            "gregsanders": "1002_csi",
            "det_evans": "1003_csi",
            "royceharmon": "1004_csi",
            "mandy": "1005_csi",
            "walterbanglor": "1006_csi",
            "bum": "1007_csi",
            "disco_placid": "1008_csi",
            "paulmillander": "1009_csi",
            "catherine": "1010_csi",
            "brass": "1011_csi",
            "sara": "1012_csi",
            "warrick": "1013_csi",
            "nick": "1014_csi",
            "grissom": "1015_csi",
        }
        spk_dict = {
            "1001_csi": "paigeharmon",
            "1002_csi": "gregsanders",
            "1003_csi": "det_evans",
            "1004_csi": "royceharmon",
            "1005_csi": "mandy",
            "1006_csi": "walterbanglor",
            "1007_csi": "bum",
            "1008_csi": "disco_placid",
            "1009_csi": "paulmillander",
            "1010_csi": "catherine",
            "1011_csi": "brass",
            "1012_csi": "sara",
            "1013_csi": "warrick",
            "1014_csi": "nick",
            "1015_csi": "grissom",
        }

    elif season == "s01e19":
        dict_spk = {
            "paulafrancis": "1001_csi",
            "lasvegaspostreporterwoman": "1002_csi",
            "needrafenway": "1003_csi",
            "nick": "1004_csi",
            "dralbertrobbins": "1005_csi",
            "tyleranderson": "1006_csi",
            "bradlewis": "1007_csi",
            "warrick": "1008_csi",
            "brass": "1009_csi",
            "sara": "1010_csi",
            "steveanderson": "1011_csi",
            "gwenanderson": "1012_csi",
            "catherine": "1013_csi",
            "grissom": "1014_csi",
        }
        spk_dict = {
            "1001_csi": "paulafrancis",
            "1002_csi": "lasvegaspostreporterwoman",
            "1003_csi": "needrafenway",
            "1004_csi": "nick",
            "1005_csi": "dralbertrobbins",
            "1006_csi": "tyleranderson",
            "1007_csi": "bradlewis",
            "1008_csi": "warrick",
            "1009_csi": "brass",
            "1010_csi": "sara",
            "1011_csi": "steveanderson",
            "1012_csi": "gwenanderson",
            "1013_csi": "catherine",
            "1014_csi": "grissom",
        }

    elif season == "s01e20":
        dict_spk = {
            "randypainter": "1001_csi",
            "sgtoriley": "1002_csi",
            "bobbydawson": "1003_csi",
            "mrsclemonds": "1004_csi",
            "adamwalkey": "1005_csi",
            "markrucker": "1006_csi",
            "bradkendall": "1007_csi",
            "dralbertrobbins": "1008_csi",
            "brass": "1009_csi",
            "janegilbert": "1010_csi",
            "sara": "1011_csi",
            "nick": "1012_csi",
            "warrick": "1013_csi",
            "catherine": "1014_csi",
            "grissom": "1015_csi",
        }
        spk_dict = {
            "1001_csi": "randypainter",
            "1002_csi": "sgtoriley",
            "1003_csi": "bobbydawson",
            "1004_csi": "mrsclemonds",
            "1005_csi": "adamwalkey",
            "1006_csi": "markrucker",
            "1007_csi": "bradkendall",
            "1008_csi": "dralbertrobbins",
            "1009_csi": "brass",
            "1010_csi": "janegilbert",
            "1011_csi": "sara",
            "1012_csi": "nick",
            "1013_csi": "warrick",
            "1014_csi": "catherine",
            "1015_csi": "grissom",
        }

    elif season == "s02e01":
        dict_spk = {
            "bonnieritten": "1001_csi",
            "nick": "1002_csi",
            "greg": "1003_csi",
            "robbins": "1004_csi",
            "sara": "1005_csi",
            "warrick": "1006_csi",
            "curtritten": "1007_csi",
            "waltbraun": "1008_csi",
            "sambraun": "1009_csi",
            "janinehaywood": "1010_csi",
            "brass": "1011_csi",
            "catherine": "1012_csi",
            "grissom": "1013_csi",
        }
        spk_dict = {
            "1001_csi": "bonnieritten",
            "1002_csi": "nick",
            "1003_csi": "greg",
            "1004_csi": "robbins",
            "1005_csi": "sara",
            "1006_csi": "warrick",
            "1007_csi": "curtritten",
            "1008_csi": "waltbraun",
            "1009_csi": "sambraun",
            "1010_csi": "janinehaywood",
            "1011_csi": "brass",
            "1012_csi": "catherine",
            "1013_csi": "grissom",
        }

    elif season == "s02e04":
        dict_spk = {
            "mrfram": "1001_csi",
            "detoriley": "1002_csi",
            "davidphillips": "1003_csi",
            "robbins": "1004_csi",
            "kelseyfram": "1005_csi",
            "dennisfram": "1006_csi",
            "managerofromaninis": "1007_csi",
            "juliabarett": "1008_csi",
            "nick": "1009_csi",
            "brass": "1010_csi",
            "sara": "1011_csi",
            "warrick": "1012_csi",
            "catherine": "1013_csi",
            "grissom": "1014_csi",
        }
        spk_dict = {
            "1001_csi": "mrfram",
            "1002_csi": "detoriley",
            "1003_csi": "davidphillips",
            "1004_csi": "robbins",
            "1005_csi": "kelseyfram",
            "1006_csi": "dennisfram",
            "1007_csi": "managerofromaninis",
            "1008_csi": "juliabarett",
            "1009_csi": "nick",
            "1010_csi": "brass",
            "1011_csi": "sara",
            "1012_csi": "warrick",
            "1013_csi": "catherine",
            "1014_csi": "grissom",
        }

    return dict_spk, spk_dict


def build_graph(source, conv, speaker, name, episode):
    """
    Build a graph from a dataframe containing the id of conversations and the speakers involved
        - source: source dataframe
        - conv: conversation column
        - speaker: speaker column
    """

    # Graph we visualize
    G = Network(notebook=True, height="500px", width="100%")
    dict_weight = {}

    # Graph we build
    G_nx = nx.Graph()

    for conversation in np.unique(source[conv]):
        sub_df = source[source[conv] == conversation]
        list_spk = np.unique(sub_df[speaker])

        for elem in list(itertools.combinations(list_spk, 2)):

            if elem[0] not in G.nodes:
                G.add_node(elem[0], label=elem[0])
            if elem[1] not in G.nodes:
                G.add_node(elem[1], label=elem[1])

            try:
                dict_weight[(elem[0], elem[1])] += 1
            except KeyError:
                dict_weight[(elem[0], elem[1])] = 1

            G_nx.add_node(elem[0])
            G_nx.nodes()[elem[0]]["name"] = elem[0]
            G_nx.add_node(elem[1])
            G_nx.nodes()[elem[1]]["name"] = elem[1]
            G_nx.add_edge(elem[0], elem[1])

            try:
                G_nx[elem[0]][elem[1]]["weight"] += 1
            except KeyError:
                G_nx[elem[0]][elem[1]]["weight"] = 1

    for conversation in np.unique(source[conv]):
        sub_df = source[source[conv] == conversation]
        list_spk = np.unique(sub_df[speaker])

        for elem in list(itertools.combinations(list_spk, 2)):

            G.add_edge(elem[0], elem[1], value=dict_weight[(elem[0], elem[1])])

    return G_nx, G.show("generated_graph/%s/%s.html" %(episode, name))


def get_all_pred_scores(file, spk_dict):
    """
    Get all the predicted scores from a file in a formated dataframe
    """

    f = open(file, "r")

    lines = []

    for line in f:
        sp = line.split()
        name_1 = spk_dict[sp[0]]
        name = sp[1]
        name_2 = sp[1].split("_Conv")[0]
        conv = sp[1].split("_Conv")[1]
        score = sp[2]
        lines.append([name_1, name, name_2, conv, score])

    pred = pd.DataFrame(lines)
    pred.columns = ["Model", "File", "Truth", "Conv", "Score"]
    pred = pred.sort_values(by=["Conv", "Truth"])
    pred["Score"] = pred["Score"].astype(float)

    return pred


def get_pred_speakers(pred):
    """
    Get all the predicted scores from a file in a formated dataframe
    """

    winners = []
    for conv in pred[["Conv", "Truth"]].drop_duplicates().iterrows():

        sub_df = pred[
            (pred["Conv"] == conv[1]["Conv"]) & (pred["Truth"] == conv[1]["Truth"])
        ]
        max_score = max(sub_df["Score"])
        line = sub_df[sub_df["Score"] == max_score]
        winners.append(
            [line["Model"].values[0], line["Truth"].values[0], line["Conv"].values[0]]
        )

    winners = pd.DataFrame(winners)
    winners.columns = ["Pred", "Truth", "Conv"]
    return winners


def speaker_accuracy(winners):
    """
    Compute speaker accuracy based on the dataframe from the speaker id output
    """
    return len(winners[winners["Pred"] == winners["Truth"]]) / len(winners)


def compute_network_score(G, G_pred):
    """
    Help function to compute a network score
    """

    score_achieved = 0
    score_max = 0
    centrality = nx.degree_centrality(G)

    for k in G.nodes():

        edges = G.edges(k)
        cent = centrality[k]

        edges_truth = G.edges(k)

        try:
            edges_pred = G_pred.edges(k)

            for edge in edges_truth:

                if edge in edges_pred:
                    score_achieved += cent
                    score_max += cent
                else:
                    score_max += cent
        except:
            score_max += cent

    return score_achieved / score_max


def compute_edge_acc(G, G_pred):
    """
    Help function to compute an edge accuracy
    """

    score_achieved = 0
    score_max = 0

    for k in G.nodes():

        edges = G.edges(k)
        edges_truth = G.edges(k)

        try:
            edges_pred = G_pred.edges(k)

            for edge in edges_truth:

                if edge in edges_pred:
                    score_achieved += 1
                    score_max += 1
                else:
                    score_max += 1
        except:
            score_max += 1

    return score_achieved / score_max


def build_candidates(pred):
    """
    Building candidates dataframe from the prediction of the speaker id
    """

    candidates = []
    for conv in np.unique(pred["Conv"]):
        sub_df = pred[pred["Conv"] == conv]
        sub_df = sub_df[sub_df["Score"] > -40]
        num_char = len(np.unique(sub_df["Truth"].values))
        candidates.append(
            [
                conv,
                num_char,
                np.unique(str(conv) + "_" + sub_df["Truth"].values)[0],
                np.unique(sub_df["Truth"].values),
                sub_df["Model"].values,
                sub_df["Score"].values,
            ]
        )

    cand = pd.DataFrame(candidates)
    cand.columns = ["Conv", "NumChar", "Conversation", "Truth", "Candidate", "Score"]
    cand["Conv"] = cand["Conv"].astype(int)
    cand = cand.sort_values(by="Conv")

    return cand


def keep_higher_scores(pred, threshold=-15):
    """
    Keep only the highest scores above a threshold
    """

    score_sup = pred[pred["Score"] >= threshold]
    score_sup["Conv"] = score_sup["Conv"].astype(int)
    score_sup = score_sup.sort_values(by=["Conv", "Truth"])
    score_sup = score_sup.reset_index().drop(["index"], axis=1)
    return score_sup


def final_speaker_accuracy(df_res, column):
    """
    Compute the final speaker accuracy from the result dataframe
    """

    acc = 0
    count = 0
    for val in df_res.iterrows():
        predic = sorted(val[1][column])
        truth = sorted(val[1]["Truth"])
        count += len(truth)
        acc += len(set(predic) & set(truth))

    return acc / count


def conversation_accuracy(df_res, column):
    """
    Compte the conversation accuracy from the result dataframe
    """

    return sum(df_res[column] == df_res["Truth"]) / len(df_res)


def rerank_graph(score_sup, winners, cand, threshold):
    """
    Core function to re-rank based on the graph knowledge
    - score_sup: dataframe above threshold
    - winners: winners dataframe
    - cand: candidates dataframe
    - threshold: starting threshold, should be same than score_sup threshold
    """

    G_rank = nx.Graph()

    trace_conv = {}
    all_comb_conv = []

    all_conv = []
    to_keep = []

    list_spk = []
    conv_list = []

    for conv in np.unique(score_sup["Conv"]):

        print("Conversation %s out of %s" % (conv, max(np.unique(score_sup["Conv"]))))

        sub_df = score_sup[score_sup["Conv"] == conv]

        all_candidate_list = []
        all_truth_list = []

        for truth in np.unique(sub_df["Truth"]):

            candidate_list = []
            truth_list = []

            sub_df2 = sub_df[sub_df["Truth"] == truth]
            candidate_list = []

            for elem in sub_df2["Model"]:
                candidate_list.append(elem)
                truth_list.append(truth)

            all_candidate_list.append(candidate_list)

        iter_conv = list(itertools.product(*all_candidate_list))
        all_comb = []

        score_speak_dict = {}

        list_truth = np.unique(sub_df["Truth"])

        for val in list_truth:
            score_speak_dict[val] = {}
            score_speak_dict[val]["name"] = ""
            score_speak_dict[val]["score"] = 0

        max_conv = threshold
        max_spk = ""

        for combination in iter_conv:

            spk_info = []
            k = 0

            for elem in combination:

                sub_df2 = sub_df[sub_df["Truth"] == list_truth[k]]
                sub_df2 = sub_df2[sub_df2["Model"] == elem]
                score_speak = max(sub_df2["Score"].values)

                try:
                    degree_speak = G_rank.degree[elem] / sum(
                        dict(G_rank.degree).values()
                    )
                except KeyError:
                    degree_speak = 0

                score_speak = score_speak * (1 + degree_speak)
                spk_info.append([elem, score_speak, degree_speak])

                if score_speak >= score_speak_dict[list_truth[k]]["score"]:
                    score_speak_dict[list_truth[k]]["score"] = score_speak
                    score_speak_dict[list_truth[k]]["name"] = elem

                k += 1

            all_comb.append([conv, spk_info])

            score_pair = 0
            who = []

            for val in spk_info:
                who.append(val[0])
                score_pair += val[1]

            for elem in list(itertools.combinations(who, 2)):
                try:
                    score_pair = score_pair * (
                        1 + trace_conv[str(sorted(elem))] / sum(trace_conv.values())
                    )
                except KeyError:
                    pass

            # try:
            # score_pair = score_pair * (1 + trace_conv[str(sorted(who))]/sum(trace_conv.values()))
            # except KeyError:
            # pass

            if score_pair > max_conv:
                max_conv = score_pair
                max_spk = combination

        for elem in list(itertools.combinations(max_spk, 2)):
            G_rank.add_edge(elem[0], elem[1])

        for elem in list(itertools.combinations(max_spk, 2)):
            if elem[0] != elem[1]:
                try:
                    trace_conv[str(sorted(elem))] += 1
                except KeyError:
                    trace_conv[str(sorted(elem))] = 1

        # try :
        # trace_conv[str(sorted(max_spk))] += 1
        # except KeyError:
        # trace_conv[str(sorted(max_spk))] = 1

        if len(all_comb) > 0:
            conv_list.append(conv)
            all_conv.append(all_comb)
            list_spk.append(list(sorted(max_spk)))

    winners["Conv"] = winners["Conv"].astype(int)
    winners = winners.sort_values(by="Conv")

    list_all = []

    for u in np.unique(winners["Conv"]):
        if u in conv_list:
            sub_df = winners[winners["Conv"] == u]
            list_guys = []
            for val in sub_df["Pred"]:
                list_guys.append(val)
            list_all.append(sorted(list_guys))

    df_res = pd.DataFrame(
        [
            np.unique(score_sup["Conv"]),
            list_spk,
            list(cand[cand["Conv"].isin(conv_list)]["Truth"].apply(lambda x: list(x))),
            list_all,
        ]
    ).T
    df_res.columns = ["Conv", "GaphEnhance", "Truth", "Prediction"]
    df_res["GaphEnhance"] = df_res["GaphEnhance"].apply(lambda x: sorted(x))
    df_res["Truth"] = df_res["Truth"].apply(lambda x: sorted(x))
    df_res["Prediction"] = df_res["Prediction"].apply(lambda x: sorted(x))

    return df_res, G_rank, trace_conv


def final_graph(G_rank, trace_conv, episode):

    G = Network(notebook=True, height="500px", width="100%")

    for edg in G_rank.edges():

        edg = sorted(list(edg))

        if edg[0] not in G.nodes:
            G.add_node(edg[0], label=edg[0])
        if edg[1] not in G.nodes:
            G.add_node(edg[1], label=edg[1])

        try:
            try:
                G.add_edge(
                    edg[0],
                    edg[1],
                    value=trace_conv[str(edg).replace("(", "[").replace(")", "]")],
                )
            except KeyError:
                G.add_edge(
                    edg[0],
                    edg[1],
                    value=trace_conv[str(edg).replace("(", "[").replace(")", "]")],
                )

        except KeyError:
            continue

    return G.show("generated_graph/%s/rerank.html"%episode)
