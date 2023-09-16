# Neural Ratio Estimation for EoR Parameter Inference (Paper In Prep)

In this study, we explore the application of a novel simulation-based inference (SBI) technique called neural ratio estimation (NRE) in the field of EoR (Epoch of Reionization) parameter inference. NRE enables the estimation of likelihood-evidence ratios, which reduces the complexity of neural networks involved in the inference process. Additionally, NRE is closely connected to representation learning and has the capability to generate informative data summaries during training.

We applied the NRE method to infer the distributions of two astrophysical parameters relevant to reionization. Our results demonstrate that NRE yields consistent outcomes compared to previous SBI techniques (see [3D ScatterNet](https://arxiv.org/abs/2307.09530), [21cmDELFI](https://github.com/Xiaosheng-Zhao/DELFI-3DCNN)). Furthermore, we show that the performance of NRE can be significantly improved by replacing pre-defined summary statistics with representations learned by the method itself.

### Methodology

As shown in many previous works(e.g., [Thomas et al 2016](https://arxiv.org/abs/1611.10242), [Miller et al 2021](https://arxiv.org/abs/2107.01214), [Durkan et al 2020](https://arxiv.org/abs/2002.03712)), we can train a classifier using Binary Cross Entropy (BCE) loss to estimate the likelihood-evidence ratio:

$$
r(\boldsymbol{\theta},\mathbf{x})=\frac{p(\boldsymbol{\theta},\mathbf{x})}{p(\mathbf{x}) p(\boldsymbol{\theta})}=\frac{p(\mathbf{x} \mid \boldsymbol{\theta})}{p(\mathbf{x})}
   %=\frac{p(\boldsymbol{\theta} \mid \mathbf{x})}{p(\boldsymbol{\theta})}
$$

With a well-trained ratio estimator, we can perform Bayesian inference by simply sampling from:

$$
p(\boldsymbol{\theta} \mid \mathbf{x})=r(\boldsymbol{\theta},\mathbf{x})p(\boldsymbol{\theta})
$$

Interestingly, the process of ratio estimation can be viewed as a representation learning procedure, allowing us to incorporate a Summary Neural Network into the framework. This additional network automatically generates informative summaries during training. For more detailed information, please refer to our paper and related works ([Chen et al 2020](https://arxiv.org/abs/2010.10079), [Hjelm et al 2019](https://arxiv.org/abs/1808.06670))

### Experiments

In our work, we apply the NRE method to infer two reionization parameters using three-dimensional images of the 21 cm signal. We provide a detailed demonstration of this process in the [experiments.ipynb](https://github.com/suicee/21cmNRE/blob/main/Experiments.ipynb) notebook. Our experiments consider mock SKA observations with thermal noise, and we also compare two different data summaries: scattering transform and NN-based summaries. For additional experiments and a comparison with DELFI results, please refer to our forthcoming paper.

For detailed implementation of the model and training procedure, please consult the [tools folder](https://github.com/suicee/21cmNRE/tree/main/tools). If you have any ideas, comments, or questions, please feel free to reach out to Ce Sui at [suic20@mails.tsinghua.edu.cn](mailto:suic20@mails.tsinghua.edu.cn).
