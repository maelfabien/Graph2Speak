#!/usr/bin/python3

# General
import sys
import numpy as np
import pandas as pd
import click

# Set of functions
from utils import *

@click.command()
@click.argument('episode', default='s01e07')
def main(episode):

    # Spedific to episode
    dict_spk, spk_dict, spk_coord = ep_dicts(episode)

    f = open("src/speaker_id_input/%s.txt"%episode, "r")
    list_spk_keep = []

    for line in f:
        list_spk_keep.append(line.replace("\n", "").replace(".", "").replace("'", ""))

    # I. Ground Truth
    print("1. Building ground truth network")
    truth_events = pd.read_csv("src/graph_input/all_events_%s.csv"%episode).drop_duplicates()

    dict_len = {}
    for c in np.unique(truth_events['conv']):
        try:
            dict_len[int(c)] = len(truth_events[truth_events['conv']==c])
        except:
            pass
        
    truth_events = truth_events[['speaker', 'conv']].dropna()
    truth_events['speaker'] = truth_events['speaker'].apply(lambda x: x.replace("/", "").replace(".", "").replace("'", ""))
    truth_events = truth_events[truth_events['speaker'].isin(list_spk_keep)]
    G, plot = build_graph(truth_events, "conv", "speaker", "truth", episode, spk_coord)
    print("Ground truth network saved in src/generated_graph/%s/truth.html"%episode)
    print(" ")

    # II. Speaker Identification Score
    print("2. Building network from speaker identification")
    pred = get_all_pred_scores("src/speaker_id_output/scores_%s/csi_test_unique_scores"%episode, spk_dict)
    winners = get_pred_speakers(pred)
    G_pred, plot_pred = build_graph(winners, "Conv", "Pred", "pred", episode, spk_coord)
    print("Speaker accuracy of the SID system is: ", speaker_accuracy(winners))
    print("Predicted network saved in src/generated_graph/%s/pred.html"%episode)
    print(" ")

    # III. Graph2Speak
    print("3. Building network from Graph2Speak")
    cand = build_candidates(pred)
    score_sup = keep_higher_scores(pred, threshold=-15)
    df_res, G_rank, trace_conv = rerank_graph(score_sup, winners, cand, dict_len, threshold=-15)
    df_res.to_csv("src/graph2speak_output/%s/output_table.csv"%episode)
    print("Results dataframe saved in src/graph2speak_output/%s/output_table.csv"%episode)

    print("---")
    print("Conversation accuracy of the SID system: ", conversation_accuracy(df_res, "Prediction"))
    print("Conversation accuracy of Graph2Speak: ", conversation_accuracy(df_res, "GaphEnhance"))
    print("---")
    print("Speaker accuracy of the SID system: ", final_speaker_accuracy(df_res, "Prediction"))
    print("Speaker accuracy of Graph2Speak: ", final_speaker_accuracy(df_res, "GaphEnhance"))
    print("---")
    plot_rank = final_graph(G_rank, trace_conv, episode, spk_coord)
    print("Graph2Speak network saved in src/generated_graph/%s/rerank.html"%episode)

    print("Different predictions between SID and Graph2Speak:")
    df_diff = df_res[df_res['GaphEnhance'] != df_res['Prediction']]
    df_diff.to_csv("src/graph2speak_output/%s/diff.csv"%(episode))
    print("Difference dataframe generated in src/graph2speak_output/%s/diff.csv"%episode)


if __name__ == "__main__":
    main()