import numpy as np
from scipy.io import wavfile
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import librosa
import plac


def main():
    """
    Takes an input episode and creates the necessary files for speaker identification in Kaldi
    """

    # List op conversations in the various episodes
    list_scenes_1_7 = [
        0,
        47,
        108,
        459,
        488,
        504,
        539,
        620,
        654,
        671,
        766,
        820,
        825,
        834,
        869,
        879,
        895,
        999,
        1074,
        1097,
        1118,
        1192,
        1230,
        1339,
        1394,
        1473,
        1487,
        1500,
        1529,
        1568,
        1649,
        1685,
        1692,
        1700,
        1750,
        1768,
        1793,
        1817,
        1851,
        1952,
        1978,
        2002,
        2035,
        2079,
        2120,
        2235,
        2284,
        2298,
        2455,
        2487,
    ]

    list_scenes_1_8 = [
        0,
        56,
        157,
        179,
        204,
        235,
        291,
        373,
        441,
        503,
        536,
        559,
        617,
        668,
        734,
        806,
        878,
        923,
        995,
        1143,
        1178,
        1222,
        1329,
        1375,
        1416,
        1452,
        1641,
        1694,
        1731,
        1765,
        1789,
        1822,
        1944,
        2021,
        2063,
        2121,
        2167,
        2218,
        2315,
        2325,
        2343,
        2420,
        2496,
        2529,
    ]

    list_scenes_1_13 = [0]

    list_scenes_1_19 = [
        0,
        85,
        2 * 60 + 5,
        2 * 60 + 35,
        3 * 60 + 7,
        4 * 60 + 6,
        5 * 60 + 8,
        5 * 60 + 42,
        5 * 60 + 55,
        6 * 60 + 14,
        8 * 60 + 1,
        8 * 60 + 54,
        10 * 60 + 15,
        10 * 60 + 59,
        11 * 60 + 53,
        12 * 60 + 57,
        13 * 60 + 11,
        13 * 60 + 34,
        14 * 60 + 2,
        15 * 60 + 2,
        15 * 60 + 18,
        15 * 60 + 33,
        15 * 60 + 56,
        16 * 60 + 17,
        16 * 60 + 20,
        16 * 60 + 57,
        17 * 60 + 42,
        18 * 60 + 3,
        19 * 60 + 34,
        21 * 60 + 14,
        21 * 60 + 58,
        22 * 60 + 51,
        24 * 60 + 15,
        24 * 60 + 24,
        25 * 60 + 30,
        25 * 60 + 34,
        25 * 60 + 39,
        25 * 60 + 44,
        25 * 60 + 47,
        26 * 22 + 22,
        26 * 60 + 34,
        27 * 60 + 39,
        28 * 60 + 24,
        28 * 60 + 48,
        28 * 60 + 58,
        31 * 60 + 18,
        31 * 60 + 35,
        31 * 60 + 37,
        32 * 60 + 40,
        33 * 60 + 55,
        34 * 60 + 4,
        38 * 60 + 17,
        38 * 60 + 46,
        41 * 60 + 15,
        42 * 60 + 26,
        42 * 60 + 44,
        50 * 60,
    ]

    list_scenes_1_20 = [
        0,
        29,
        52,
        57,
        60 + 6,
        60 + 51,
        2 * 60 + 21,
        2 * 60 + 21,
        3 * 60 + 57,
        4 * 60 + 10,
        5 * 60 + 31,
        5 * 60 + 56,
        6 * 60 + 24,
        8 * 60 + 6,
        9 * 60,
        9 * 60 + 27,
        9 * 60 + 35,
        9 * 60 + 42,
        10 * 60 + 23,
        12 * 60 + 4,
        12 * 60 + 32,
        13 * 60 + 43,
        14 * 60 + 21,
        14 * 60 + 45,
        14 * 60 + 58,
        15 * 60 + 35,
        17 * 60 + 17,
        18 * 60 + 39,
        19 * 60 + 48,
        20 * 60 + 6,
        20 * 60 + 10,
        20 * 60 + 17,
        20 * 60 + 37,
        20 * 60 + 45,
        21 * 60 + 18,
        22 * 60 + 5,
        22 * 60 + 36,
        23 * 60,
        23 * 60 + 15,
        23 * 60 + 26,
        24 * 60 + 2,
        24 * 60 + 34,
        25 * 60 + 3,
        26 * 60 + 26,
        26 * 60 + 30,
        29 * 60 + 45,
        29 * 60 + 54,
        30 * 60 + 11,
        30 * 60 + 21,
        30 * 60 + 41,
        30 * 60 + 58,
        31 * 60 + 40,
        32 * 60 + 34,
        32 * 60 + 57,
        33 * 60 + 29,
        34 * 60 + 17,
        34 * 60 + 54,
        35 * 60 + 58,
        36 * 60 + 16,
        36 * 60 + 55,
        37 * 60 + 10,
        37 * 60 + 51,
        39 * 60 + 41,
        40 * 60 + 22,
        40 * 60 + 41,
        40 * 60 + 51,
        41 * 60 + 7,
        41 * 60 + 37,
        42 * 60 + 40,
        50 * 60,
    ]

    list_scenes_2_1 = [
        0,
        91,
        2 * 60 + 28,
        3 * 60 + 31,
        4 * 60 + 1,
        4 * 60 + 18,
        5 * 60 + 22,
        5 * 60 + 58,
        6 * 60 + 41,
        7 * 60 + 21,
        7 * 60 + 53,
        9 * 60 + 27,
        9 * 60 + 47,
        10 * 60 + 11,
        10 * 60 + 55,
        11 * 60 + 18,
        12 * 60 + 1,
        12 * 60 + 11,
        14 * 60 + 21,
        15 * 60 + 48,
        16 * 60,
        16 * 60 + 12,
        16 * 60 + 58,
        17 * 60 + 24,
        18 * 60 + 16,
        18 * 60 + 53,
        19 * 60 + 47,
        19 * 60 + 52,
        20 * 60 + 52,
        22 * 60 + 40,
        23 * 60 + 39,
        23 * 60 + 45,
        24 * 60 + 52,
        26 * 60 + 20,
        27 * 60 + 29,
        28 * 60 + 32,
        28 * 60 + 54,
        29 * 60 + 4,
        29 * 60 + 12.3,
        29 * 60 + 43,
        30 * 60 + 21,
        31 * 60 + 6.5,
        31 * 60 + 26,
        31 * 60 + 35,
        32 * 60 + 42,
        34 * 60 + 2,
        35 * 60 + 7.5,
        36 * 60 + 20,
        36 * 60 + 50,
        37 * 60 + 44,
        37 * 60 + 53,
        38 * 60 + 53,
        40 * 60 + 4,
        40 * 60 + 21,
        40 * 60 + 37,
        41 * 60 + 42,
        50 * 60,
    ]

    list_scenes_1_23 = [
        0,
        55,
        60 + 51.5,
        2 * 60 + 12,
        2 * 60 + 41,
        3 * 60 + 19,
        4 * 60 + 7,
        4 * 60 + 33,
        5 * 60 + 26,
        6 * 60 + 10,
        6 * 60 + 15.5,
        7 * 60 + 15,
        7 * 60 + 57,
        8 * 60 + 25,
        8 * 60 + 44,
        9 * 60 + 25,
        9 * 60 + 47,
        12 * 60 + 22,
        14 * 60 + 22,
        14 * 60 + 59,
        15 * 60 + 51,
        16 * 60 + 5,
        16 * 60 + 50,
        17 * 60 + 43,
        18 * 60 + 30,
        18 * 60 + 57,
        19 * 60 + 4.5,
        19 * 60 + 6.5,
        19 * 60 + 45.5,
        19 * 60 + 50,
        20 * 60 + 34,
        21 * 60 + 48,
        22 * 60 + 20,
        24 * 60 + 22,
        25 * 60 + 4,
        25 * 60 + 18.5,
        26 * 60 + 7,
        26 * 60 + 28.5,
        26 * 60 + 36.5,
        27 * 60 + 20,
        27 * 60 + 49,
        30 * 60 + 17,
        30 * 60 + 47,
        31 * 60 + 16,
        32 * 60 + 50,
        34 * 60,
        34 * 60 + 30,
        35 * 60,
        35 * 60 + 14,
        40 * 60 + 25,
        40 * 60 + 54,
        40 * 60 + 59,
        41 * 60 + 45,
        42 * 60 + 10,
        42 * 60 + 32,
        42 * 60 + 49,
        43 * 60 + 21,
    ]

    list_scenes_2_4 = [
        0,
        44,
        88,
        60 + 58,
        2 * 60 + 33,
        2 * 60 + 42,
        3 * 60 + 14,
        3 * 60 + 34,
        5 * 60 + 22,
        6 * 60 + 2,
        6 * 60 + 41,
        7 * 60 + 36,
        7 * 60 + 56,
        8 * 60 + 59,
        9 * 60 + 36,
        9 * 60 + 44,
        10 * 60 + 46,
        11 * 60 + 6,
        11 * 60 + 38,
        13 * 60,
        13 * 60 + 19,
        14 * 60 + 13,
        14 * 60 + 49,
        16 * 60 + 6,
        17 * 60 + 19,
        17 * 60 + 33,
        17 * 60 + 39,
        18 * 60 + 25,
        18 * 60 + 43,
        19 * 60 + 6.5,
        19 * 60 + 18,
        19 * 60 + 26,
        20 * 60 + 9,
        20 * 60 + 56,
        21 * 60 + 56.6,
        22 * 60 + 13.5,
        23 * 60 + 1,
        23 * 60 + 54,
        24 * 60 + 16,
        25 * 60 + 28,
        26 * 60 + 42,
        27 * 60 + 9,
        27 * 60 + 18,
        28 * 60 + 44,
        29 * 60 + 4,
        29 * 60 + 24,
        29 * 60 + 31,
        30 * 60 + 2,
        30 * 60 + 32,
        30 * 60 + 48,
        31 * 60 + 54,
        32 * 60 + 15,
        33 * 60 + 30,
        33 * 60 + 32,
        34 * 60 + 29,
        35 * 60 + 35,
        35 * 60 + 50,
        35 * 60 + 57,
        36 * 60 + 24,
        38 * 60 + 44,
        39 * 60 + 44,
        50 * 60,
    ]

    list_scenes_2_6 = [
        0,
        58,
        67, 
        90,
        119,
        2*60+25,
        3*60+32,
        3*60+45.5,
        4*60+18,
        4*60+34.2,
        4*60+44,
        5*60+8,
        6*60+3,
        6*60+34,
        7*60+21,
        7*60+45,
        9*60+5,
        9*60+43,
        11*60,
        11*60+47,
        12*60+10,
        12*60+57,
        13*60+50,
        14*60+19,
        15*60+30,
        16*60+16,
        18*60+34,
        19*60+12.8,
        20*60+2,
        22*60+14,
        22*60+46,
        23*60+17,
        23*60+29,
        24*60+5,
        25*60+15,
        25*60+30,
        25*60+35,
        25*60+46,
        25*60+49,
        25*60+52.5,
        26*60+3,
        26*60+36,
        26*60+53,
        27*60+52,
        29*60+16,
        29*60+55,
        30*60+23,
        31*60+9,
        31*60+50,
        31*60+57,
        32*60+9,
        34*60+25,
        34*60+54,
        35*60+53,
        36*60+25,
        36*60+50,
        37*60+37,
        37*60+52,
        40*60+39,
        41*60+19,
        42*60+14,
        42*60+42  
    ]
        
    list_scenes = {
        "s01e07": list_scenes_1_7,
        "s01e08": list_scenes_1_8,
        "s01e19": list_scenes_1_19,
        "s01e20": list_scenes_1_20,
        "s01e23": list_scenes_1_23,
        "s02e01": list_scenes_2_1,
        "s02e04": list_scenes_2_4,
        "s02e06": list_scenes_2_6,
    }

    def timestamp_to_float(timest):
        return datetime.timestamp(timest) - datetime.timestamp(
            datetime.strptime("1900-01-01 00:00:00.000000", "%Y-%m-%d %H:%M:%S.%f")
        )

    def fill_col(x):
        if len(x) == len("00:02:32"):
            return str(x) + ".000000"
        else:
            return x

    def split_scenes(start, end):

        for s in range(len(list_scenes_ep)):
            if start >= list_scenes_ep[s] and end <= list_scenes_ep[s + 1]:
                return int(s)
                break

    def split_wav(name, start, end, rate, x):

        for s in range(len(list_scenes_ep)):
            if start >= list_scenes_ep[s] and end <= list_scenes_ep[s + 1]:
                wavfile.write(
                    "src/data/clean_out/"
                    + ep
                    + "/"
                    + name.replace("/", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("_", "")
                    + "_"
                    + "Conv"
                    + str(s)
                    + "_"
                    + str(start)
                    + "_"
                    + str(end)
                    + ".wav",
                    rate,
                    x[int(float(start) * rate) : int(float(rate) * end)],
                )
                break

    def split_per_speaker(name, rate, x):

        sub_df = df_clean[df_clean["speaker"] == name]
        data_utt = []

        for line in sub_df.iterrows():

            if line[1]["speaker"] == name:

                data1 = x[
                    int(float(line[1]["start_time_float"]) * rate) : int(
                        float(line[1]["end_time_float"]) * rate
                    )
                ]
                if len(data_utt) == 0:
                    data_utt = data1
                else:
                    data_utt = np.concatenate((data_utt, data1))

        wavfile.write(
            "src/data/all_concatenated/%s/" % ep
            + name.replace("/", "").replace("(", "").replace(")", "").replace("_", "")
            + ".wav",
            rate,
            data_utt,
        )

        return int(len(data_utt) / rate)

    for ep in ["s01e07", "s01e08", "s01e19", "s01e20", "s01e23", "s02e01", "s02e04", "s02e06"]:

        print("Processing episode : %s" % ep)

        list_scenes_ep = list_scenes[ep]

        # Split the raw WAV files into individual files, per speaker per scene
        fs, data = wavfile.read("src/data/wav_episodes/%s.wav" % ep)

        df = pd.read_csv("src/data/tsv/%s.tsv" % ep, sep="\t")
        df = df[df["speaker"] != "None"]
        first = df["speaker"].values[0]

        df["end_time"] = pd.to_datetime(
            df["end_time"].apply(lambda x: fill_col(x)), format="%H:%M:%S.%f"
        )
        df["start_time"] = pd.to_datetime(
            df["start_time"].apply(lambda x: fill_col(x)), format="%H:%M:%S.%f"
        )
        df["start_time_float"] = df["start_time"].apply(lambda x: timestamp_to_float(x))
        df["end_time_float"] = df["end_time"].apply(lambda x: timestamp_to_float(x))

        df["conv"] = df.apply(
            lambda x: split_scenes(x["start_time_float"], x["end_time_float"]), axis=1
        )
        df["utt_id"] = df["speaker"] != df["speaker"].shift(1)
        df["utt_id"] = df["utt_id"].cumsum()

        df.to_csv("src/graph_input/all_events_%s.csv" % ep, index=False)

        utt_df = []

        for spk in np.unique(df["utt_id"]):
            utt_df.append(
                [
                    np.unique(df[df["utt_id"] == spk]["speaker"])[0],
                    spk,
                    min(df[df["utt_id"] == spk]["start_time_float"]),
                    max(df[df["utt_id"] == spk]["end_time_float"]),
                ]
            )

        utt_df = pd.DataFrame(
            utt_df, columns=["Speaker", "UtteranceID", "Start", "Stop"]
        )
        utt_df["utt_length"] = utt_df["Stop"] - utt_df["Start"]

        df.apply(
            lambda x: split_wav(
                x["speaker"], x["start_time_float"], x["end_time_float"], fs, data
            ),
            axis=1,
        )

        df_clean = df[
            ["speaker", "start_time_float", "end_time_float", "conv"]
        ].drop_duplicates()

        len_audio = []

        for speaker in np.unique(df_clean["speaker"]):
            len_audio.append([speaker, split_per_speaker(speaker, fs, data)])
        len_audio = np.array(len_audio)

        speaker_times = pd.DataFrame(len_audio)
        speaker_times.columns = ["Speaker", "Time"]
        speaker_times["Time"] = speaker_times["Time"].astype(int)
        speaker_times = speaker_times.sort_values(by="Time")

        speakers_to_keep = speaker_times[speaker_times["Time"] > 20]
        f = open("src/speaker_id_input/%s.txt" % ep, "w")
        for spk in np.unique(speakers_to_keep["Speaker"]):
            f.write(spk + "\n")
        f.close()

        name_vs_utt = []

        for name in speakers_to_keep["Speaker"]:

            sub_df = df_clean[df_clean["speaker"] == name]
            data_utt = []

            for line in sub_df.iterrows():

                if line[1]["speaker"] == name:

                    data1 = data[
                        int(float(line[1]["start_time_float"]) * fs) : int(
                            float(line[1]["end_time_float"]) * fs
                        )
                    ]
                    if len(data_utt) == 0:
                        data_utt = data1
                    else:
                        data_utt = np.concatenate((data_utt, data1))

            # If we have more than 50 seconds we can afford to split
            if len(data_utt) > 50 * fs:
                wavfile.write(
                    "src/speaker_id_input/enroll_files/%s/" % ep
                    + name.replace("/", "")
                    .replace(".", "")
                    .replace("'", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("_", "")
                    + ".wav",
                    fs,
                    data_utt[: 40 * fs],
                )
                name_vs_utt.append(
                    [
                        name.replace("/", "")
                        .replace(".", "")
                        .replace("'", "")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("_", ""),
                        name.replace("/", "")
                        .replace(".", "")
                        .replace("'", "")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("_", "")
                        + ".wav",
                    ]
                )
            else:
                wavfile.write(
                    "src/speaker_id_input/enroll_files/%s/" % ep
                    + name.replace("/", "")
                    .replace(".", "")
                    .replace("'", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("_", "")
                    + ".wav",
                    fs,
                    data_utt,
                )
                name_vs_utt.append(
                    [
                        name.replace("/", "")
                        .replace(".", "")
                        .replace("'", "")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("_", ""),
                        name.replace("/", "")
                        .replace(".", "")
                        .replace("'", "")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("_", "")
                        + ".wav",
                    ]
                )

        df2 = pd.DataFrame(name_vs_utt)
        df2.columns = ["Name", "File"]
        df2["channel"] = "a"
        df2.to_csv("src/speaker_id_input/train_file_%s.tsv" % ep, sep="\t", index=False)

        spk_vs_file = []

        for conv in np.unique(df_clean["conv"]):

            sub_df = df_clean[df_clean["conv"] == conv]
            sub_df = sub_df[
                ["speaker", "start_time_float", "end_time_float"]
            ].drop_duplicates()

            for name in np.unique(sub_df["speaker"]):

                if name in list(speakers_to_keep["Speaker"]):

                    data_utt = []
                    sub_df2 = sub_df[sub_df["speaker"] == name]

                    for line in sub_df2.iterrows():

                        data1 = data[
                            int(float(line[1]["start_time_float"]) * fs) : int(
                                float(line[1]["end_time_float"]) * fs
                            )
                        ]

                        if len(data_utt) == 0:
                            data_utt = data1
                        else:
                            data_utt = np.concatenate((data_utt, data1))

                    if len(data_utt) > 5000:
                        spk_vs_file.append(
                            [
                                name.replace("/", "")
                                .replace(".", "")
                                .replace("'", "")
                                .replace("(", "")
                                .replace(")", "")
                                .replace("_", ""),
                                name.replace("/", "")
                                .replace(".", "")
                                .replace("'", "")
                                .replace("(", "")
                                .replace(")", "")
                                .replace("_", "")
                                + "_Conv"
                                + str(int(conv)),
                            ]
                        )
                        wavfile.write(
                            "src/speaker_id_input/test_files/%s/" % ep
                            + name.replace("/", "")
                            .replace(".", "")
                            .replace("'", "")
                            .replace("(", "")
                            .replace(")", "")
                            .replace("_", "")
                            + "_Conv"
                            + str(int(conv))
                            + ".wav",
                            fs,
                            data_utt,
                        )

        f = open("src/speaker_id_input/test_files/transcript_%s.txt" % ep, "w")

        for conv in np.unique(df["conv"]):

            sub_df = df[df["conv"] == conv]
            sub_df = sub_df[
                ["speaker", "word", "start_time_float", "end_time_float"]
            ].drop_duplicates()

            for name in np.unique(sub_df["speaker"]):

                if name in list(speakers_to_keep["Speaker"]):

                    data_utt = []
                    sub_df2 = sub_df[sub_df["speaker"] == name]
                    sent = ""

                    for line in sub_df2.iterrows():

                        data1 = data[
                            int(float(line[1]["start_time_float"]) * fs) : int(
                                float(line[1]["end_time_float"]) * fs
                            )
                        ]

                        if len(data_utt) == 0:
                            data_utt = data1
                            sent = line[1]["word"]
                        else:
                            data_utt = np.concatenate((data_utt, data1))
                            if line[1]["word"][0] == "'":
                                sent = sent + line[1]["word"]
                            else:
                                sent = sent + " " + line[1]["word"]

                    if len(data_utt) > 5000:
                        f.write(name + "; " + str(int(conv)) + "; " + sent + "\n")

        f.close()

        df3 = pd.DataFrame(spk_vs_file)
        df3.columns = ["Name", "File"]
        df3["channel"] = "a"
        df3.to_csv("src/speaker_id_input/test_file_%s.tsv" % ep, sep="\t", index=False)


plac.call(main)
