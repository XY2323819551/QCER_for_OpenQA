from reranking.utils.rider import rider_rerank_pyserini
import json
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path-data', type=str, default=None, help='path to retrieval res to rerank')
parser.add_argument('--path-pred', type=str, default=None, help=' path to npred file ')
parser.add_argument('--path-out', type=str, default=None, help='path to save rerank results')
parser.add_argument('--fusion', type=str, default=None, help='fusion parameters that we chosen')
parser.add_argument('--n-contexts', type=int, default=1000, help="sort the first n passages")
parser.add_argument('--topk-em', type=int, default=10, help="the reader predictions based on topk-em passages")
parser.add_argument('--n-pred', type=int, default=1, help='the number of reader predictions for rerenking')
parser.add_argument('--save-res', default=True, action='store_true')
args = parser.parse_args()


def load_data(path_data, path_pred):
    with open(path_data, "r", encoding="utf-8") as f2:
        data2 = json.load(f2)
    with open(path_pred, "r", encoding="UTF-8") as f_pred:
        q2pred = json.load(f_pred)
    return data2, q2pred


if __name__ == "__main__":
    data, q2pred = load_data(args.path_data, args.path_pred)
    rider_rerank_pyserini(data, q2pred, args.n_contexts, args.n_pred, args, key='contexts_rerank_100psg')
    print("Finished!")
