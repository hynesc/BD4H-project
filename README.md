## Project Description:

In this project, we replicate Xiao et al. (2018), who proposed the CONTENT model for hospital readmission prediction and claimed it outperforms state-of-the-art baselines. However, we find that the reported performance gains are not statistically significant. We reimplemented the CONTENT model and the GRU benchmark in PyTorch and evaluated them on the same synthetic dataset. Our results show no significant difference in performance between the two under the original setup. To extend the work, we introduced a grid search for hyperparameter tuning. With this optimization, CONTENT does outperform the baselines, supporting the model’s potential under improved training conditions.

## Instructions for Execution:

We created this project in Google Colab. To execute this, simply download this github repo and upload it unzipped to your drive. In the code folder, click and open the jupyter file "CONTENT_colab.ipynb" in Google Colab. The Table of Contents will help guide you through what we did step-by-step. The outputs you see after each code block are our implementation results. If you would like to run it yourself, make sure to connect to a GPU runtime and hit run all!

Note: you will also see files get_embedding.py and content-env.yaml in the code folder. Since Google Colab does not support Gensim, we needed to run these locally to generate our Word2Vec matrix. You do not need to run these, as the matrix is already saved to the resource folder, which the CONTENT_colab.ipynb automatically pulls from :)
