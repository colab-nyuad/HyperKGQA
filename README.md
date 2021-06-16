## HyperbolicKGQA
HyperKGQA,  proposes a technique that embeds a Knowledge Graph into the hyperbolic space  and  leverages  this  pre-trained  embeddings  to  map  questions' representation  into entities  and  relationships  space. An extensive set of experiments was run on two benchmark datasets using code published in this repository. The results show that the proposed  method  performs  significantly  better  than the state-of-the-art tehcniqiues, especially when reasoning is on arbitrary multi-hop questions over large sparse graphs embedded into a low-dimensional representation.

### Installation
```sh
# retrieve and install project in development mode
git clone https://github.com/colab-nyuad/Hyperbolic_KGQA.git
cd kge
pip install -e .

# set environment variables
cd ..
source set_env.sh
```

### Datasets

### Usage
To train and evaluate a QA task over KG, use the main.py script:

```sh
usage: main.py [-h] [--dataset {MetaQA,fbwq}]
              [--model {TransE,RESCAL,CP,Distmult,SimplE,RotH,RefH,AttH,ComplEx,RotatE}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam}]
              [--max_epochs MAX_EPOCHS] [--valid_every VALID]
              [--dim RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]

Knowledge Graph Embedding

optional arguments:
  -h, --help            show this help message and exit
  --dataset {FB15K,WN,WN18RR,FB237,YAGO3-10}
                        Knowledge Graph dataset
  --model {TransE,CP,MurE,RotE,RefE,AttE,RotH,RefH,AttH,ComplEx,RotatE}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --debug               Only use 1000 examples for debugging
  --multi_c             Multiple curvatures per relation
  ```
### Experiments

### Computing graph curvature
Two metrics are available to estimate how hierarchical relations in a KG are: the curvature estimate and the Krackhardt hierarchy score. The curvature estimate captures global hierarchical behaviours (how much the graph is tree-like) and the Krackhardt score reflects more local behaviour (how many small loops the graph has). For details on computing metrics please refer to [Low-Dimensional Hyperbolic Knowledge Graph Embeddings](https://arxiv.org/abs/2005.00545) and [Learning mixed-curvature representations in product spaces](https://openreview.net/pdf?id=HJxeWnCcF7). To compute the metrics, use the graph_curvature.py script with the following arguments:
```sh
usage: graph_curvature.py [-h] [--dataset DATASET] 
                          [--kg_type KG_TYPE] 
                          [--curvature_type {'krackhardt', 'global_curvature'}] 
                          [--relation RELATION]
```
If computing graph curvature is done before link prediction or separatly, plese make sure that the name of the dataset is according to the format <dataset>_<kg_type> and is placed inside the folder kge/data. To preprocess the dataset run the following command inside the folder data:
```
python preprocess/preprocess_default.py <dataset>_<kg_type>
```
Following is an example command to compute the Krackhardt hierarchy score for all relations in a KG: 
```
python graph_curvature.py --dataset MetaQA --kg_type half --curvature_type krackhardt
```
or for a specific relation:
```
python graph_curvature.py --dataset MetaQA --kg_type half --curvature_type krackhardt --relation _instance_hypernym
```
