# A Heterogeneous Graph to Abstract Syntax Tree Framework for Text-to-SQL

This is the repository containing source code for the TAPMI 2023 journal article [*A Heterogeneous Graph to Abstract Syntax Tree Framework for Text-to-SQL*](https://ieeexplore.ieee.org/document/10194972), HG2AST hereafter. If you find it useful, please cite our work.
```
@ARTICLE{10194972,
  author={Cao, Ruisheng and Chen, Lu and Li, Jieyu and Zhang, Hanchong and Xu, Hongshen and Zhang, Wangyou and Yu, Kai},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Heterogeneous Graph to Abstract Syntax Tree Framework for Text-to-SQL}, 
  year={2023},
  volume={45},
  number={11},
  pages={13796-13813},
  keywords={Structured Query Language;Decoding;Databases;Syntactics;Semantics;Task analysis;Computational modeling;Abstract syntax tree;grammar-based constrained decoding;heterogeneous graph neural network;knowledge-driven natural language processing;permutation invariant problem;text -to-SQL},
  doi={10.1109/TPAMI.2023.3298895}}
```

## Code Migration Declaration

> Note that: This work focuses on leveraging **small-sized** bi-directional pre-trained models (e.g., BERT, ELECTRA) and labeled training data to train a **specialized**, **interpretable** and **efficient local** text-to-SQL parser in **low-resource** scenarios, instead of chasing SOTA performances. For better results, please try LLM with in-context learning (such as [DINSQL](https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting) and [ACTSQL](https://github.com/X-LANCE/text2sql-GPT)), or resort to larger encoder-decoder architectures containing billion parameters (such as [Picard-3B](https://github.com/ServiceNow/picard) and [RESDSQL-3B](https://github.com/RUCKBReasoning/RESDSQL)). Due to a shift in the author's research focus in the LLM era, the relevant code repository will no longer be maintained.

The original code structure relies on very complicated graph-based encoding and tree-based decoding modules, which are difficult to maintain and not "elegant" enough for small-sized models. When token-based LLM decoder becomes mainstream, we believe the unique advantage of symbolic HG2AST framework is its **lightness, ease of training and inference efficiency**.

**Thus, we refine the original model structure and propose a more general grammar-based Transformer decoder, called [ASTormer](https://arxiv.org/pdf/2310.18662.pdf), as the termination of a symbolic framework based on graph encoding and tree decoding. The new code repository can be found [here](https://github.com/rhythmcao/text-to-sql-astormer).**

```
@misc{cao2023astormer,
      title={ASTormer: An AST Structure-aware Transformer Decoder for Text-to-SQL}, 
      author={Ruisheng Cao and Hanchong Zhang and Hongshen Xu and Jieyu Li and Da Ma and Lu Chen and Kai Yu},
      year={2023},
      eprint={2310.18662},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```