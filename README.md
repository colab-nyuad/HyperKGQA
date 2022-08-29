# HyperKGQA: Question Answering over Knowledge Graphs using Hyperbolic Representation Learning

HyperKGQA proposes a technique that embeds a Knowledge Graph into the hyperbolic space  and  leverages  this  pre-trained  embeddings  to  map  questions' representation  into entities  and  relationships  space. An extensive set of experiments was run on two benchmark datasets using code published in this repository. The results show that the proposed  method  performs  better  than the state-of-the-art techniques when reasoning on arbitrary multi-hop questions over large sparse graphs.

![](architecture.jpg "Overall architecture")
*HyperKGQA architecture for question answering including the (i) knowledge graph embedding, (ii) AttH question embedding, (iii) Hyperbolic Question Embedding (with an optional Relation Composition component that we evaluate separately), and (iv) Answer Selection (with path matching).*

### Quick start
```sh
# retrieve and install project in development mode
git clone https://github.com/colab-nyuad/HyperKGQA.git
cd kge
pip install -e .

# set environment variables
cd ..
source set_env.sh
```

## Table of contents
1. [Data](#data)
2. [Parameters](#usage)
3. [Computing embeddings](#emb)
4. [Run KGQA](#kgqa)
5. [Results](#results)

## Data <a name="data"></a>

The repo presents results for two QA datasets MetaQA and WebQuestionsSP. MetaQA dataset with its underlying KG can be downloaded [here](https://github.com/yuyuz/MetaQA). WebQuestionsSP dataset is available for download in json format from [here](https://www.microsoft.com/en-us/download/details.aspx?id=52763). The underlying KG for WebQuestionsSP was selected as a subset of Freebase KG with following run of Page Rank algorithm to reduce the size of the graph. For the detailed description on this and download of full and sparse versions of KG please refer to the baseline paper [Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://www.aclweb.org/anthology/2020.acl-main.412/). Please unzip KGs datasets (train, valid and test files) into <em>kg_data/dataset_name/setting</em>, where setting indicates full or half. For QA datasets files, for WebQuestionsSP please put files into the folder <em>qa_data/dataset_name</em> and for MetaQA please put files in a subfolder indicating the number of hops, e.g., <em>qa_data/MetaQA/1hop/</em>.

## Parameters <a name="usage"></a>
The main script to run KGQA task is main.py, following we provide the description of the parameters it has:

```sh
--dataset           the name of the dataset, should meatch folders created at the previous step
--hops              need to specified as int(1, 2 or 3) for MetaQA dataset
--kg_type           setting full or sparse
--model             an emdedding model (for all choices please refer to the file)
--regularizer       which regulariztion to use for KGQA
--reg               regularization weight
--optimizer         which optimizer to use (for all choices please refer to the file) 
--max_epochs        for how many epochs to train KGQA
--patience          number of epochs before early stopping
--valid_every       number of training phases before validation
--batch_size        batch size
--learning_rate     learning rate for KGQA
--freeze            freezing weights of trained embeddings
--use_cuda          use gpu
--gpu               which gpu to use
--num_workers       number of workers for dataloader
--dim               embedding dimension
--checkpoint_type   choices=['libkge', 'ldh'], depending on which library was used to compute embeddings 
--rel_gamma         hyperparameter for relation matching
```

## Computing embeddings <a name="emb"></a>
Running the script main.py computes KG embeddings using [LibKGE](https://github.com/uma-pi1/kge) and QA task over KG. To compute the embeddings using LibKGE, training parameters (learning_rate, batch_size, optimizer_type, dropout, normalization_metric and etc.) need to be specified in a config file. The script checks if there is an uploaded config file in the fomrat: \<dataset\>\_\<kg_type\>\_\<model\>\_\<dim\> in the folder kge/data/config_files/<dataset> to use for training embeddings. If the file not found, the config will be created from the input arguments. 

## Run KGQA <a name="kgqa"></a>


Following is an example command to run tarining KG embedding and QA task for sparse MetaQA dataset, dimension 200, AttH model and 1hop questions: 

```sh
python main.py --dataset MetaQA --model AttH --dim 400 --kg_type half --valid_every 5 --max_epochs 200 \
--learning_rate_kgqa 0.0002 --hops 1 --qa_nn_type LSTM
```

For Freebase:
```sh
python main.py --dataset fbwq --model ComplEx --dim 50 --kg_type full --valid_every 10 --max_epochs 200 \
--learning_rate_kgqa 0.00002 --freeze True --batch_size 16 --qa_nn_type RoBERTa
```
  
To use already pretrained embeddings, please specifiy the path to the folder with files checkpoint_best.pt, entity_ids.del and relation_ids.del:

```sh
python main.py --dataset MetaQA --embeddings data/pretrained_models/MetaQA/AttH_MetaQA_half_400/ \
--model AttH --dim 400 --kg_type half --valid_every 5 --max_epochs 200 --learning_rate_kgqa 0.0002 --hops 3 \
--qa_nn_type LSTM
```
  
Compute performance with relation matching:
```sh
python main.py --dataset fbwq --model RefH --dim 400 --kg_type half --valid_every 10 --max_epochs 200 \
--learning_rate_kgqa 0.00002 --freeze True --batch_size 16 --qa_nn_type RoBERTa --use_relation_matching True
```
  
## Results <a name="results"></a>
All results on KGQA are available in the manuscript. Please refer to the Tables 3-7. Here we present the results on Link Prediction.

For MetaQA KG:
<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Setting</th>
            <th>Dimension</th>
            <th>Model</th>	
            <th>MRR</th>	
            <th>H@1</th>	
            <th>H@3</th>
            <th>H@10</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=8>MetaQA_KG</td>
            <td rowspan=4>half</td>
            <td rowspan=2>50</td>
            <td>ComplEx	</td>
            <td>0.097</td>	
            <td>0.069</td>	
            <td>0.108</td>
            <td>0.148</td>
        </tr>
        <tr>
          <td>AttH</td>	
          <td>0.112</td>
          <td>0.072</td>	
          <td>0.123</td>
          <td>0.187</td>
        </tr>
        <tr>
            <td rowspan=2>400</td>
            <td>ComplEx	</td>
            <td>0.123</td>	
            <td>0.09</td>	
            <td>0.133</td>
            <td>0.185</td>
        </tr>
        <tr>
          <td>AttH</td>	
          <td>0.159</td>
          <td>0.112</td>	
          <td>0.176</td>
          <td>0.249</td>
        </tr>
        <tr>
            <td rowspan=4>full</td>
            <td rowspan=2>50</td>
            <td>ComplEx	</td>
            <td>0.99</td>	
            <td>0.981</td>	
            <td>0.999</td>
            <td>0.1</td>
        </tr>
          <td>AttH</td>	
          <td>0.917</td>
          <td>0.889</td>	
          <td>0.937</td>
          <td>0.96</td>
        </tr>
          <tr>
            <td rowspan=2>400</td>
            <td>ComplEx	</td>
            <td>1</td>	
            <td>1</td>	
            <td>1</td>
            <td>1</td>
        </tr>
          <td>AttH</td>	
          <td>0.992</td>
          <td>0.985</td>	
          <td>0.999</td>
          <td>1</td>
        </tr>
    </tbody>
</table>

For WebQuestionsSP KG:
<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Setting</th>
            <th>Dimension</th>
            <th>Model</th>	
            <th>MRR</th>	
            <th>H@1</th>	
            <th>H@3</th>
            <th>H@10</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=8>WQSP_KG</td>
            <td rowspan=4>half</td>
            <td rowspan=2>50</td>
            <td>ComplEx	</td>
            <td>0.531</td>	
            <td>0.494</td>	
            <td>0.553</td>
            <td>0.6</td>
        </tr>
        <tr>
          <td>AttH</td>	
          <td>0.579</td>
          <td>0.539</td>	
          <td>0.602</td>
          <td>0.651</td>
        </tr>
        <tr>
            <td rowspan=2>400</td>
            <td>ComplEx	</td>
            <td>0.594</td>	
            <td>0.567</td>	
            <td>0.608</td>
            <td>0.646</td>
        </tr>
        <tr>
          <td>AttH</td>	
          <td>0.585</td>
          <td>0.541</td>	
          <td>0.609</td>
          <td>0.663</td>
        </tr>
        <tr>
            <td rowspan=4>full</td>
            <td rowspan=2>50</td>
            <td>ComplEx	</td>
            <td>0.895</td>	
            <td>0.844</td>	
            <td>0.937</td>
            <td>0.992</td>
        </tr>
          <td>AttH</td>	
          <td>0.852</td>
          <td>0.817</td>	
          <td>0.869</td>
          <td>0.918</td>
        </tr>
          <tr>
            <td rowspan=2>400</td>
            <td>ComplEx	</td>
            <td>0.98</td>	
            <td>0.965</td>	
            <td>0.996</td>
            <td>0.999</td>
        </tr>
          <td>AttH</td>	
          <td>0.85</td>
          <td>0.819</td>	
          <td>0.866</td>
          <td>0.908</td>
        </tr>
    </tbody>
</table>

## How to cite
