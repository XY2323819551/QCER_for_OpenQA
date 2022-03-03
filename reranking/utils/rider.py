import copy
import json
import unicodedata
from tqdm import tqdm
from .tokenizers import SimpleTokenizer

simple_tokenizer = SimpleTokenizer()


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def has_answer(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == 'string':
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    return False


def calc_pyserini_retrieval_acc(data, k, ctx_key="contexts"):
    ct = 0
    for d in data:
        for ctx in d[ctx_key][:k]:
            if ctx["has_answer"]:
                ct += 1
                break
    return ct / len(data)


def rider_rerank_pyserini(data, q2pred, n_contexts, n_pred, args, key='contexts_rerank_100psg'):
    data = [data[str(i)] for i in range(len(q2pred))]
    for d in tqdm(data):
        d[key] = []
        for context in d['contexts'][:n_contexts]:
            # if contains reader predictions, add to new contexts list
            pred = q2pred[d['question']]
            has_answer_flag = has_answer(answers=set(pred[args.fusion][str(args.topk_em)][:n_pred]), text=context['text'].replace("\n", " "),
                                         tokenizer=simple_tokenizer, match_type='string')
            if has_answer_flag:
                context_new = copy.deepcopy(context)
                context_new['has_answer'] = has_answer(answers=d['answers'], text=context['text'].replace("\n", " "),
                                                       tokenizer=simple_tokenizer, match_type='string')
                d[key].append(context_new)

        id_set = set([context['docid'] for context in d[key]])
        for context in d['contexts'][:n_contexts]:
            if context['docid'] not in id_set:
                d[key].append(copy.deepcopy(context))

    # print('\t\t old   rerank')
    data_res = {}
    for k in [1, 5, 10, 20, 100, 500, 1000]:
        acc = calc_pyserini_retrieval_acc(data, k=k)
        acc_rerank = calc_pyserini_retrieval_acc(data, k=k, ctx_key=key)
        data_res[k] = [acc, acc_rerank]
        # print(f'top-{k} acc:\t {acc:.4f} {acc_rerank:.4f}')

    data_new = {}
    for i2, d in enumerate(tqdm(data)):
        d["contexts"] = d[key]
        d.pop(key)
        data_new[str(i2)] = d

    if args.save_res:
        with open(args.path_out + ".json", "w", encoding="utf-8") as f_out1:
            json.dump(data_res, f_out1, indent=4)
        with open(args.path_out + ".retrieval.res" + ".json", "w", encoding="utf-8") as f_out2:
            json.dump(data_new, f_out2, indent=4)

    # if path_out is not None:
    #     write_rerank_psg(data, path_out, ctx_key="contexts_rerank_100psg")

    # if save_res:
    #     for d in tqdm(data):
    #         d['contexts'] = d['contexts_rerank_100psg']
    #         d.pop('contexts_rerank_100psg')
    #     with open(path_out, "w", encoding="utf-8") as f_out:
    #         json.dump(data, f_out, indent=4)


# 根据传入的ctx_key来分别进行处理
def write_rerank_psg(data, fname, ctx_key="ctxs_rerank_100psg"):
    """
    write reranked passages to file as the input of the generative reader
    """
    with open(fname, "w") as o:
        for d in data:
            s = d["question"] + " </s> "
            for idx in range(10):
                if ctx_key == "ctxs_rerank_100psg":
                    s += (
                            d[ctx_key][idx]["title"]
                            + " </s> "
                            + d[ctx_key][idx]["text"]
                            + " </s> "
                    )

                elif ctx_key == "contexts_rerank_100psg":
                    s += (
                            d[ctx_key][idx]["text"].replace("\n", "</s>")
                            + " </s> "
                    )
            o.write(s + "\n")


def gar_has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    # Answer is a list of possible strings
    text = tokenizer.tokenize(text).words(uncased=True)

    for single_answer in answers:
        single_answer = _normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)

        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return True
