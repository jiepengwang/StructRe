# StructRe: Rewriting for Structured Shape Modeling
In this paper, we propose a robust and generalizable approach for structured shape modeling, by using local and probabilistic structure rewriting. The rewriting system transits locally between two levels of part decompositions within a hierarchy, therefore getting rid of categorical templates and generalizing across categories. The rewriting system also allows probabilistic sampling of different transitions that are equally plausible, therefore resolving the ambiguity with diverse structures. Specifically, we first build a unified latent space of all parts and shapes across categories. Given the latent encoding, we learn the rewriting model that maps a set of input parts into another set of output parts, with iterative decoding at the output side to sample multiple solutions according to probability.


![](./doc/teaser.png)

## [Project page](https://jiepengwang.github.io/StructRe/) |  [Paper](https://arxiv.org/abs/2311.17510)  |  [Data](https:***)

Code is coming soon...

## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{wang2023structre,
    title={StructRe: Rewriting for Structured Shape Modeling}, 
    author={Wang, Jiepeng and Pan, Hao and Liu, Yang and Tong, Xin and Komura, Taku and Wang, Wenping},
    journal={arXiv preprint arXiv:2311.17510},
    year={2023}
}
```