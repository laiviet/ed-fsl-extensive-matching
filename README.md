# Extensively matching

Extensively Matching for Few-shot Learning Event Detection

https://arxiv.org/abs/2006.10093


## Data

I couldnt share my preprocessed dataset, please preprocess the data yourself with the following features

```
indices: Glove/Word2vec indices of the sentence (B,N,K+Q,L)
dist: indices of the relative position embedding (B,N,K+Q,L)
length: length of sentence  (B,N,K+Q,1)
mask: mask (0 mean padding tokens, 1 otherwise) (B,N,K+Q,L)
anchor_index: index of the trigger/anchor word (B,N,K+Q,1)

where 
B is batch size, 
N is number of classes, 
K+Q is the number of sample in the support set and query set per class
L is the max_sentence length
```

## Citation

```
@inproceedings{lai-etal-2020-extensively,
    title = "Extensively Matching for Few-shot Learning Event Detection",
    author = "Lai, Viet Dac  and
      Nguyen, Thien Huu  and
      Dernoncourt, Frank",
    booktitle = "Proceedings of the First Joint Workshop on Narrative Understanding, Storylines, and Events",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.nuse-1.5",
    doi = "10.18653/v1/2020.nuse-1.5",
    pages = "38--45"
}
```

## Acknowledgement:

This source code is built based on the framework released by @thunlp at https://github.com/thunlp/FewRel


