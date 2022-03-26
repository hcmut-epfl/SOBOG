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
