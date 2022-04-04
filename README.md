# Explanation Quality Guided Path Reasoning (EQGPR)
This is the code for the paper "Reinforcement Recommendation Reasoning through Knowledge Graphs for Explanation Quality" currently proceeding in the Elseiver Special Issue "Knowledge-Graph-Enabled Artificial Intelligence (KG-enabled AI)".

[Visual Abstract]()

## Datasets
Three public avaiable datasets from different domains, ML1M for movies, LASTFM for songs and Amazon Beauty for eCommerce.

Since Amazon eCommerce datasets are formatted in the same way, more Amazon eCommerce datasets can be preprocessed and used (refer to dataset_mapper.py).

The original datasets without preprocessing can be found here: [ML1M]() [1], [LASTFM1B]() [2], [Beauty]() [3].

The external knowledge (KGs) used can be found here: [ML1M]() [4], [LASTFM1B]() [5], [Beauty]() [6].

We preprocessed the datasets following the procedures reported in the manuscript. The preprocessed datasets can be downloaded here: []().

## Requirements
- Python >= 3.6
You can install the other requirements using: 
```
pip install -r requirements.txt
```

## Usage

### In-Train

1. Proprocess the data first:
```bash
python preprocess.py --dataset <dataset_name>
```
"<dataset_name>" should be one of "ml1m", "lastfm", "beauty" (refer to utils.py).

2. Train knowledge graph embeddings (TransE in this case):
```bash
python train_transe_model.py --dataset <dataset_name>
```

3. Train RL agent with the optimized path-selecting strategy and scoring:
```bash
python train_agent.py --dataset <dataset_name> --metric <metric_to_consider_in_train> --alpha <weight_for_the_metric>
```

4. Produce the paths with the optimized beam search:
```bash
python test_agent.py --dataset <dataset_name> --metric <metric_to_consider_in_train> --alpha <weight_for_the_metric>
```

5. Predicted paths will be avaiable in the paths folder, they can be evaluated through the main.py with the ```eval_baseline=True``` or performe the Post-Processing previously proposed. (Refer to the readme [here](https://github.com/giacoballoccu/PPE-Path-Reasoning-RecSys/blob/main/README.md))

# References
\[1\] Post Processing Recommender Systems with Knowledge Graphs
for Recency, Popularity, and Diversity of Explanations
\[2\] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, and Yongfeng Zhang. 2019. Reinforcement knowledge graph reasoning for explainable recommendation. In Proceedings of the 42nd International ACM SIGIR (Paris, France) https://github.com/orcax/PGPR 
