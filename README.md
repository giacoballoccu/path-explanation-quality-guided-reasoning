# Explanation Quality Guided Path Reasoning (EQGPR)
This is the code for the paper "Reinforcement Recommendation Reasoning through Knowledge Graphs for Explanation Quality" currently proceeding in the Elseiver Special Issue "Knowledge-Graph-Enabled Artificial Intelligence (KG-enabled AI)".

![Visual Abstract](/KG-EAI-vabstract.png)

The other baselines are located in the other repository: [Knowlede-Aware-Recommender-Systems-Baselines](https://github.com/giacoballoccu/Knowlede-Aware-Recommender-Systems-Baselines)
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

## Computational Complexity
Under our post-processing approach, once the platform decides to include reasoning path quality in the recommendation policy, the LIR (SEP) values for each user (product), for all interactions, should be computed. 
Computing all LIR values has $\mathcal{O}(|U| * |T_u|)$ complexity based on the number of users $|\mathcal U|$ and of interactions per user $| T_u |$, with in general $| T_u | << |\mathcal U|$. Computing the LIR value for a new user interaction has $O(1)$ complexity. 
Conversely, computing all SEP values has a complexity in terms of number of entity types $|\lambda|$ and number of entities per entity type $| E_{\lambda} |$, that is $\mathcal{O}(|\lambda| * | E_{\lambda} |)$, in general $| \lambda | < < | E_{\lambda} |$. The insertion of a new product, interaction, or entity type would require to (re)compute the SEP values for the entities with the type that product belongs to. We however assume that SEP values will be updated periodically (and not after every insertion), given that the popularity patterns would not change in the short period. Furthermore, the computation for both LIR and SEP can be easily parallelized. 

Optimization on PTD does not require any pre-computation. Therefore, given the above pre-computed matrices, our post-processing optimization for a user on LIR, SEP or PTD has complexity $\mathcal{O}(m\log{}m)$, with $m = | \mathcal{L}_{u}|$ being the number of predicted paths for user $u$ by the original model.  

# References
\[1\] Post Processing Recommender Systems with Knowledge Graphs
for Recency, Popularity, and Diversity of Explanations

\[2\] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, and Yongfeng Zhang. 2019. Reinforcement knowledge graph reasoning for explainable recommendation. In Proceedings of the 42nd International ACM SIGIR (Paris, France) https://github.com/orcax/PGPR 
