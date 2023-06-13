import utils
import torch
import numpy as np
from tqdm import tqdm
from model.SOBOG import SOBOG
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split
from data.dataloader import TwitterDataset, SecondTwitterDataset
from data.twidataloader import Twibot22SampleDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score

parser = ArgumentParser(description="SOBOG")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--enable_gpu", type=bool, default=False)
parser.add_argument("--n_user_features", type=int, default=15)
parser.add_argument("--d_user_embed", type=int, default=50)
parser.add_argument("--n_post_features", type=int, default=384)
parser.add_argument("--d_post_embed", type=int, default=100)
parser.add_argument("--n_gat_layers", type=int, default=1)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--train_size", type=float, default=0.6)
parser.add_argument("--d_post_cls", type=int, default=32)
parser.add_argument("--n_post_cls_layer", type=int, default=2)
parser.add_argument("--d_user_cls", type=int, default=16)
parser.add_argument("--n_user_cls_layer", type=int, default=4)
parser.add_argument("--num_edge_type", type=int, default=2)
parser.add_argument("--trans_head", type=int, default=2)
parser.add_argument("--semantic_head", type=int, default=2)
parser.add_argument("--user_embed_dropout", type=float, default=0.3)
parser.add_argument("--path", type=str, default='/content/twibot_22_sample_test_text-sbert_normalized.pt')
args = parser.parse_args()

n_post_features = args.n_post_features
n_user_features = args.n_user_features

if __name__ == "__main__":
    dataset = torch.load(args.path)

    n_users = len(dataset)
    train_size = args.train_size
    train_length = int(n_users * train_size)
    test_length = n_users - train_length

    train_set, test_set = random_split(
        dataset,
        [train_length, test_length],
        torch.manual_seed(0)
    )
    del (dataset)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn_padd
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn_padd
    )

    # Initialize the model
    # if args.n_gpu > 0:
    #     cmd = utils.set_cuda_visible_device(args.n_gpu)
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]
    arg_dict = utils.parse_to_dict(args)
    model = SOBOG(gpu=1 if args.enable_gpu else 0, **arg_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.enable_gpu else "cpu")
    model = utils.initialize_model(model, device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss function
    loss_user_fn = torch.nn.BCELoss()
    # loss_tweet_fn = utils.tweet_loss_fn
    loss_tweet_fn = torch.nn.BCELoss()

    train_losses_epoch = list()
    test_losses_epoch = list()

    for epoch in range(args.epoch):
        print("Epoch %d/%d".format(epoch + 1, args.epoch))

        user_losses, tweet_losses = list(), list()
        train_losses, test_losses = list(), list()
        train_true, test_true = list(), list()
        train_pred, test_pred = list(), list()

        model.train()
        for iter, sample in enumerate(tqdm(train_loader)):
            model.zero_grad()
            user, tweet, adj, up, label, tlabel = sample
            user, tweet, adj, up = user.to(device), tweet.to(device), adj.to(device), up.to(device)
            label, tlabel = label.to(device), tlabel.to(device)

            user_pred, tweet_pred = model.forward(user, tweet, adj, up)

            user_loss = loss_user_fn(user_pred, label)
            tweet_loss = loss_tweet_fn(tweet_pred, tlabel)
            total_loss = user_loss * args.alpha + tweet_loss * (1 - args.alpha)

            total_loss.backward()
            optimizer.step()

            # Collect loss, true label and predicted label

            user_losses.append(user_loss.data.cpu().numpy())
            tweet_losses.append(tweet_loss.data.cpu().numpy())
            train_losses.append(total_loss.data.cpu().numpy())
            train_true.append(user_pred.data.cpu().numpy())
            train_pred.append(label.data.cpu().numpy())

            user, tweet, adj, up = user.to("cpu"), tweet.to("cpu"), adj.to("cpu"), up.to("cpu")
            label, tlabel = label.to("cpu"), tlabel.to("cpu")
            if iter % 10 == 9:
                user_acc = round(
                    (np.round(np.array(train_true)) == np.array(train_pred)).sum() / (args.batch_size * (iter + 1)), 4)
                mean_loss = round(np.array(train_losses).mean(), 4)
                user_loss_mean = round(np.array(user_losses).mean(), 4)
                tweet_loss_mean = round(np.array(tweet_losses).mean(), 4)
                print(" loss:", mean_loss, "- acc:", str(user_acc), "- user_loss:", user_loss_mean, "- tweet_loss:",
                      tweet_loss_mean)
        train_losses_epoch.append(np.array(train_losses).mean())
        torch.save(model.state_dict(), "model/model.pt")
        model.eval()
        for sample in tqdm(test_loader):
            model.zero_grad()

            user, tweet, adj, up, label, tlabel = sample
            user, tweet, adj, up = user.to(device), tweet.to(device), adj.to(device), up.to(device)
            label, tlabel = label.to(device), tlabel.to(device)

            user_pred, tweet_pred = model.forward(user, tweet, adj, up)

            test_pred.append(user_pred.data.cpu().numpy())
            test_true.append(label.data.cpu().numpy())

            user, tweet, adj, up = user.to("cpu"), tweet.to("cpu"), adj.to("cpu"), up.to("cpu")
            label, tlabel = label.to("cpu"), tlabel.to("cpu")

        test_pred_flattened = np.concatenate(test_pred).ravel()
        test_true_flattened = np.concatenate(test_true).ravel()

        print(accuracy_score(test_true_flattened, np.round(test_pred_flattened)))
        print(precision_score(test_true_flattened, np.round(test_pred_flattened)))
        print(recall_score(test_true_flattened, np.round(test_pred_flattened)))
