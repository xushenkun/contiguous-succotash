# Pytorch Recurrent Variational Autoencoder with Dilated Convolutions

## Model:
This is the implementation of Zichao Yang's [Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/abs/1702.08139)
with Kim's [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615) embedding for tokens

![model_image](images/model_image.png)

Most of the implementations about the recurrent variational autoencoder are adapted from [analvikingur/pytorch_RVAE](https://github.com/analvikingur/pytorch_RVAE)

## Usage
### Before model training it is necessary to train word embeddings:
```
$ python train_word_embeddings.py
```

This script train word embeddings defined in [Mikolov et al. Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546)

#### Parameters:
`--use-cuda`

`--num-iterations`

`--batch-size`

`--num-sample` –– number of sampled from noise tokens


### To train model use:
```
$ python train.py
```

#### Parameters:
`--use-cuda`

`--num-iterations`

`--batch-size`

`--learning-rate`
 
`--dropout` –– probability of units to be zeroed in decoder input

`--use-trained` –– use trained before model

### To sample data after training use:
```
$ python sample.py
```
#### Parameters:
`--use-cuda`

`--num-sample`

nohup ../python3/bin/python3 -u train.py &
../python3/bin/python3 -u sample.py --use-trained ./data/poem_trained_2100_RVAE --beam-size 50 --z-size 30