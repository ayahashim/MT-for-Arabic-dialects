# MT-for-Arabic-dialects

We explore two approaches to improve machine translation for low-resource Arabic dialects: unsupervised domain adaptation and backtranslation. Arabic dialects exhibit distinct linguistic features, with some dialects being more similar to each other than others. Leveraging this characteristic, we demonstrate that a model trained on one group of dialects can effectively translate other dialects without additional labelled data. This approach leads to improved translation quality for all dialects and reduces the gap between distinct and similar dialects. Our proposed methodology involves initially training a neural machine translation model on various dialects using parallel corpora. Subsequently, fine-tuning is performed on unlabelled data of another dialect, where the translation model is jointly trained with an unsupervised domain adaptation discriminator. We also show that backtranslation can improve the performance of a base model, by generating synthetic parallel data and selecting sentences similar in domain to those in the existing parallel corpus. Domain cosine and domain fine-tune methods are deployed using different language models, to select data from the generated parallel data. Finally, we show that a multi-dialect model utilizing both unsupervised domain adaptation and backtranslation can outperform all other versions of our models and also those from the literature.

The objective of this project is to improve machine translation for unlabelled Arabic dialects to modern standard Arabic, using unsupervised domain adaptation and backtranslation techniques. In particular, we want to:

    1- improve the translation of the most distinctive Arabic dialect (the North Africa dialect) using an unsupervised domain adaptation method;
    2- investigate several data selection methods for backtranslation;
    3- deploy backtranslation to improve the performance of a translation model;
    4- compare multi-dialect with single-dialect models;
    5- benchmark our proposed models to models from the literature;
    6- combine the unsupervised domain adaptation technique with backtranslation and perform ablation on these components.
