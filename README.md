# QCER_for_OpenQA
# 	Query Context Expansion Retrieval (QCER)

This repo provides the code and resources of QCER.

Humans are accustomed to associating the prior knowledge with the text in a query autonomously when doing question-answering, but for a machine that lack cognition and common sense, the query is just a simple sentence of some words. Although we can enrich semantic information of query through language representation or query expansion, the information contained in the query is still not enough. So we propose an effective passage retrieval method named Query Context Expansion Retrieval (QCER) for Open-Domain Question Answering (OpenQA), which associates the query with the domain information by adding contextual association information to the query. We implement QCER by appending reader predictions (theoretically present in candidate paragraphs) as the contextual information of the query to the initial query. QCER with sparse representations (BM25) can improve retrieval efficiency and accelerate the convergence of query, so that reader can find the desired answer as soon as possible by using fewer relevant passages. Moreover, QCER can be easily combined with DPR to achieve even better performance as sparse and dense representations are often complementary. Remarkably, we demonstrated that QCER achieves state-of-the-art performance for three tasks of passage retrieval, reading, and reranking on Natural Questions and TriviaQA datasets under the extractive QA setup.

## Installation

We provide  `environment.yml`. You can use  `conda env create -f environment.yml && conda activate envname`(envname is on the first line of the environment.yml file, ours name is QCER)

## Data

- Most of our code comes from [Pyserini](https://github.com/XY2323819551/pyserini/blob/master/docs/experiments-dpr.md), [Pygaggle](https://github.com/castorini/pygaggle/blob/master/docs/experiments-dpr-reader.md) and [GAR](https://github.com/morningmoni/GAR). 
- We now provide the [query-context-expanded queries](https://pan.baidu.com/s/1WIDSJG1HKOoACn-aE_8kAw ) (Extracted Code: QCER)(in case you wonder what they look like and/or perform new retrieval yourself) so that you can retrieve the new retrieval results by your self. You may achieve performance improvements by using the new retrieval results of **OCER** and **OCER+DPR** during inference. 
- Before conduction retrieval, you must download [resources](https://pan.baidu.com/s/1NWPhm52m8gGgD7qQkzTzwA) (Extracted Code: QCER) to QCER/pyserini/
- Experiment results are all based on test set of NQ and Trivia.

## Code

### Retrieval

**DPR retrieval** with brute-force index:

```
$ python -m pyserini.dsearch --topics the path to initial query or expanded query \
                             --index wikipedia-dpr-multi-bf \
                             --encoder facebook/dpr-question_encoder-multiset-base \
                             --output runs/file-name.trec \
                             --batch-size 36 --threads 12
```

**BM25 retrieval**

```
$ python -m pyserini.search --topics the path to initial query or expanded query \
                            --index wikipedia-dpr \
                            --output runs/file-name.trec
```

**Hybrid dense-sparse retrieval** 

```
$ python -m pyserini.hsearch dense  --index wikipedia-dpr-multi-bf \
                                    --encoder facebook/dpr-question_encoder-multiset-base \
                             sparse --index wikipedia-dpr \
                             fusion --alpha the value of fusion factor dependes on dataset \
                             run    --topics the path to initial query or expanded query \
                                    --batch-size 36 --threads 12 \
                                    --output runs/file-name.trec 
```

`--topics`: Path to queries, you can choose initial query or expanded query of NQ and Trivia.

`--index`: If you download them, you can replace them with local paths, which may be faster.

`--output`: Path to save your retrieval results.

`--fusion`: In first cycle, we set fusion factor to $1.3$ and $0.95$ for NQ and Trivia respectively. And for second cycle, we set $1.0$ and $0.95$ for both dataset respectively.



To evaluate, first convert the TREC output format to DPR's `json` format:

```
$ python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run --topics the path to initial query or expanded query \
                                                                --index wikipedia-dpr \
                                                                --input runs/file-name.trec \
                                                                --output runs/file-name.json \
                                                                --store-raw

$ python -m pyserini.eval.evaluate_dpr_retrieval --retrieval runs/file-name.json --topk 5 20 100 500 1000
```



### Reading

```
$ python -m pygaggle.run.evaluate_passage_reader --task wikipedia --retriever score --reader dpr \
                                                  --settings garfusion_beta_gamma \
                                                  --model-name only for Trivia
                                                  --retrieval-file path to save retrieval results \
                                                  --output-file Path to the results of reader inference \
                                                  --topk-em 10 20 50 100 200 500 \
                                                  --device cuda 
```

`--settings`: For sparse retrieval results, we set `garfusion_0.46_0.308` for NQ and `garfusion_0.78_0.093` for Trivia respectively. For hybrid retrieval results, we set `garfusion_0.32_0.1952` for NQ and `garfusion_0.76_0.152` for Trivia respectively.

`--model-name`: For NQ, we set  `--model-name` as `facebook/dpr-reader-single-nq-base`. For Trivia, we set `--model-name` as `facebook/dpr-reader-multiset-base`.

`--retrieval-file`: The input file of reader.

`--output-file`: Path to save the results of reader inference.

`--topk-em`: How many passage selected as the reader input.

`--device`:  cuda or cuda:1, cuda:2...

### Generating Expanded Query

```
$ python -m scripts.generate_expanded_query --path-topics Path to initial query \
                                            --data-name nq-test or Trivia-tes
                                            --path-pred Path to the results of reader inference \
                                            --path-out Path to save the expanded query \ 
                                            --settings "GAR Fusion, beta=xx, gamma=xx" \
                                            --n-pred Number of reader predictions used as expansion terms \
                                            --topk-em the reader predictions are obtained based on top-k passages
```

`--settings`: For sparse retrieval results, we set `"GAR Fusion, beta=0.46, gamma=0.308"` for NQ and `"GAR Fusion, beta=0.78, gamma=0.093"` for Trivia respectively. For hybrid retrieval results, we set `"GAR Fusion, beta=0.32, gamma=0.1952"` for NQ and `"GAR Fusion, beta=0.76, gamma=0.152"` for Trivia respectively.

### Manual Hybird

Combining the new sparse search results with the original dense search results manually. 

```
$ python -m scripts.manual_hybrid --path-sparse Path to new sparse retrieval results based on expanded query \
                                  --path-dense Path to initial dense retrieval results based on initial query \
                                  --path-topics XX Path to initial query file\
                                  --path-out Path to save the new hybrid retrieval results \
                                  --alpha 1.0 for NQ and 0.95 for Trivia
```

### Reranking

Reranking the initial retrieval results by using new reader predictions based on new retrieval results as reranking signals.

```
$ python -m reranking.rerank --path-data Path to initial retrieval results which needs to rerank \
                             --path-pred Path to the new results of reader inference based on new retrieval results \
                             --path-out Path to save reranking results \
                             --fusion "GAR Fusion, beta=xx, gamma=xx" \
                             --n-contexts sort the first n passage \
                             --topk-em the reader predictions are obtained based on top-k passages \
                             --n-pred Number of reader predictions used as reranking signals
```

`--fusion`: Same as `--setting` above, depends on which kind of retrieval results and dataset we used.

`--topk-em`: Same as `--topk-em` above, but here the top-k passages are selected from the new retrieval results.

### Example

We provide an example based on the NQ dataset that shows how to use these commands. We have written these commands in `command.md`.(We  will upload this file later.)

