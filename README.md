# Extensively matching

Extensively Matching for Few-shot Learning Event Detection

https://arxiv.org/abs/2006.10093


## Data

I couldnt share my preprocessed dataset, you can purchase the data set from LDC https://catalog.ldc.upenn.edu/LDC2006T06
please preprocess the data yourself. 

Where the function  ``utils.read_ace_data(utils.ACE)`` in this line:
https://github.com/laiviet/ed-fsl-extensive-matching/blob/a20e617f621ebd2b906983073652251bf8aa9c3b/dataloader.py#L68
should return a ``data`` dictionary and a ``label2index`` dictionary whose data formats are shown in the 

https://github.com/laiviet/ed-fsl-extensive-matching/blob/master/data.json

and 

https://github.com/laiviet/ed-fsl-extensive-matching/blob/master/label2index.json

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


