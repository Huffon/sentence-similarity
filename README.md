# Sentence Similarity
This repo contains various ways to calculate the similarity between source and target sentences. You can use **the pre-trained models** you want to use such as _ELMo_, _BERT_ and _Universal Sentence Encoder (USE)_.

And you can also choose **the method** to be used to get the similarity:

    1. Cosine similarity
    2. Euclidean distance
    3. Inner product
    3. TS-SS score
    4. Pairwise-cosine similarity
    5. Pairwise-cosine similarity + IDF


## Usage
- You have to choose the model and method to be used to calculate the similarity between source and target sentences.
- You should wrap your source and target sentences with double quotations(").

```
python sensim.py
    --source "SOURCE_SENTENCE"
    --target "TARGET_SENTENCE"
    --model  MODEL_NAME
    --method METHOD_NAME
```

## Requirements
```
allennlp==0.9.0
en-core-web-sm==2.1.0
numpy==1.17.2
spacy==2.1.8
tensorflow==2.0.0
tensorflow-hub==0.7.0
torch==1.2.0
transformers==2.1.1
```


## References
### Papers
- [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
- [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [A Hybrid Geometric Approach for Measuring Similarity Level Among Documents and Document Clustering](https://ieeexplore.ieee.org/document/7474366/metrics#metrics)

### Libraries
- [TF-hub's Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2)
- [Allen NLP's ELMo](https://github.com/allenai/allennlp)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [Vector Similarity](https://github.com/taki0112/Vector_Similarity)