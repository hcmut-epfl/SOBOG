# SOBOG

## To run the model

1. Clone the repository.
2. Download [dataset_full.pt](https://drive.google.com/file/d/1ET_PUcza8DNFLC03HrEGeVt50_RzjfMM/view?usp=sharing) and put it in the `data` folder. (If you put on other folder or want to rename the file, `path` variable is required to be specified in the script).
3. Run the script
- By default, just run:

```
python main.py
```

- For more configurations:

```
usage: main.py [-h] [--lr LR] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--enable_gpu ENABLE_GPU] [--n_user_features N_USER_FEATURES]
               [--d_user_embed D_USER_EMBED] [--n_post_features N_POST_FEATURES] [--d_post_embed D_POST_EMBED] [--n_gat_layers N_GAT_LAYERS] [--d_cls D_CLS]   
               [--n_cls_layer N_CLS_LAYER] [--alpha ALPHA] [--train_size TRAIN_SIZE] [--path PATH]
```

## To infer new data sample
1. Download [dataset_full.pt](https://drive.google.com/file/d/1ET_PUcza8DNFLC03HrEGeVt50_RzjfMM/view?usp=sharing) and put it in the `data` folder.
2. Download [vectorizer.pk](https://drive.google.com/file/d/1XVf7zs3a1joDqrDBJcdsLwlN8aBiJz6A/view?usp=sharing) and put it in the `vect` folder.
(In step 1 and 2, if you put on other folder or want to rename the file, `model_path` and `tfidf_path` variable is required to be modified in the `__init__` function of `Inference` object).
4. Take a quick look at [inference.py](https://github.com/hcmut-epfl/SOBOG/blob/experiment/inference.py). There is a `sample_user_object` and `sample_tweet_object` from line 145 that shows what input can be adapted.
5. Do one of the followings:
  - (For software team) Go to [this link](https://docs.google.com/document/d/1nSw99q2fQl4nlPRwjGq8z8Qv_ueSoIJQK6C5wuaxLcE/edit?usp=sharing) to know how we acquire the data from Twitter API. 
  - (For others) Just simply modify the `sample_user_object` and `sample_tweet_object` to see how the result changes.
6. Run the script
```
python inference.py
```

*By default, you will get a tensor with value 0.0892.*
