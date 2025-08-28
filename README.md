
## Code Repository

This is the code repository for the papers:

- [A Weak Supervision Approach for Few-Shot Aspect Based Sentiment Analysis](https://aclanthology.org/2024.eacl-long.167/) (Vacareanu et al., EACL 2024)

- [Instruction Tuning for Few-Shot Aspect-Based Sentiment Analysis](https://aclanthology.org/2023.wassa-1.3/) (Varia et al., WASSA 2023)

## Subdirectories

The repository has the following directories:

1. data: contains the necessary python scripts to create the few-shot data. The source code in the data directory is same as the source code in this repository: https://github.com/amazon-science/instruction-tuning-for-absa
2. baselines: contains the necessary python scripts to create the weak supervision data as described in the first paper above
3. instruction_tuning: contains the necessary python scripts to instruction tune the T5 model on the weak supervision data
4. utils: contains other utility python scripts

For the data, check the README inside the data directory.

If you find the sources useful, please consider citing our work:

# Citations

BibTeX for the first paper:

```
@inproceedings{vacareanu-etal-2024-weak,
    title = "A Weak Supervision Approach for Few-Shot Aspect Based Sentiment Analysis",
    author = "Vacareanu, Robert  and
      Varia, Siddharth  and
      Halder, Kishaloy  and
      Wang, Shuai  and
      Paolini, Giovanni  and
      Anna John, Neha  and
      Ballesteros, Miguel  and
      Muresan, Smaranda",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.167/",
    doi = "10.18653/v1/2024.eacl-long.167",
    pages = "2734--2752",
    abstract = "We explore how weak supervision on abundant unlabeled data can be leveraged to improve few-shot performance in aspect-based sentiment analysis (ABSA) tasks. We propose a pipeline approach to construct a noisy ABSA dataset, and we use it to adapt a pre-trained sequence-to-sequence model to the ABSA tasks. We test the resulting model on three widely used ABSA datasets, before and after fine-tuning. Our proposed method preserves the full fine-tuning performance while showing significant improvements (15.84 absolute F1) in the few-shot learning scenario for the harder tasks. In zero-shot (i.e., without fine-tuning), our method outperforms the previous state of the art on the aspect extraction sentiment classification (AESC) task and is, additionally, capable of performing the harder aspect sentiment triplet extraction (ASTE) task."
}
```

and BibTeX for the second paper:

```
@inproceedings{varia-etal-2023-instruction,
    title = "Instruction Tuning for Few-Shot Aspect-Based Sentiment Analysis",
    author = "Varia, Siddharth  and
      Wang, Shuai  and
      Halder, Kishaloy  and
      Vacareanu, Robert  and
      Ballesteros, Miguel  and
      Benajiba, Yassine  and
      Anna John, Neha  and
      Anubhai, Rishita  and
      Muresan, Smaranda  and
      Roth, Dan",
    editor = "Barnes, Jeremy  and
      De Clercq, Orph{\'e}e  and
      Klinger, Roman",
    booktitle = "Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment, {\&} Social Media Analysis",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wassa-1.3/",
    doi = "10.18653/v1/2023.wassa-1.3",
    pages = "19--27",
    abstract = "Aspect-based Sentiment Analysis (ABSA) is a fine-grained sentiment analysis task which involves four elements from user-generated texts:aspect term, aspect category, opinion term, and sentiment polarity. Most computational approaches focus on some of the ABSA sub-taskssuch as tuple (aspect term, sentiment polarity) or triplet (aspect term, opinion term, sentiment polarity) extraction using either pipeline or joint modeling approaches. Recently, generative approaches have been proposed to extract all four elements as (one or more) quadrupletsfrom text as a single task. In this work, we take a step further and propose a unified framework for solving ABSA, and the associated sub-tasksto improve the performance in few-shot scenarios. To this end, we fine-tune a T5 model with instructional prompts in a multi-task learning fashion covering all the sub-tasks, as well as the entire quadruple prediction task. In experiments with multiple benchmark datasets, we show that the proposed multi-task prompting approach brings performance boost (by absolute 8.29 F1) in the few-shot learning setting."
}
```