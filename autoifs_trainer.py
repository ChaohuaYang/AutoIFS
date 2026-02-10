import torch
import argparse
import logging
import os, sys
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils
from modules import autoifs

parser = argparse.ArgumentParser(description="AutoIFS trainer")
parser.add_argument("--dataset", type=str, help="specify dataset", default="ml-1m")
parser.add_argument("--model", type=str, help="specify model", default="autoifs")

# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate", default=3e-4)
parser.add_argument("--l2", type=float, help="L2 regularization", default=3e-5)
parser.add_argument("--r_lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("--r_l2", type=float, help="L2 regularization", default=3e-4)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--search_epoch", type=int, default=10, help="sreach epochs")
parser.add_argument("--max_epoch", type=int, default=50, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, default="save/", help="model save directory")


# neural network hyperparameters
parser.add_argument("--dim", type=int, help="embedding dimension", default=16)
parser.add_argument("--mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate (default:0.0)")
parser.add_argument("--mlp_bn", action="store_true", default=False, help="mlp batch normalization")
parser.add_argument("--cross", type=int, help="cross layer", default=3)
parser.add_argument("--domain_list", type=int, default=[0, 1, 2], help="domain_id list")
parser.add_argument("--task_num", type=int, default=2, help="domain_id list")

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=-1, help="device info")

# lora
parser.add_argument("--lora_r", type=int, help="lora rank", default=2)
parser.add_argument("--final_temp", type=float, default=100, help="final temperature")
parser.add_argument("--reg_lambda1", type=float, default=0.0, help="reg_lambda")
parser.add_argument("--reg_lambda2", type=float, default=0.0, help="reg_lambda")
parser.add_argument("--rewind", type=int, default=9, help="rewind model")

args = parser.parse_args()

my_seed = 2022
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'


class Trainer(object):
    def __init__(self, opt):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.r_lr = opt['r_lr']
        self.r_l2 = opt['r_l2']
        self.bs = opt['bsize']
        self.opt = opt
        self.model_dir = opt["save_dir"]
        self.dataset = opt["dataset"]
        self.domain_list = opt["model_opt"]['domain_list']
        self.task_num = opt["model_opt"]['task_num']
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        self.network = autoifs.AutoIFS(opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = autoifs.getOptim(self.network, opt["optimizer"], self.lr, self.l2)
        self.logger = trainUtils.get_log(opt['model'])
        self.temp_increase = opt["final_temp"] ** (1. / (opt["search_epoch"] - 1))
        self.reg_lambda1 = opt['reg_lambda1']
        self.reg_lambda2 = opt['reg_lambda2']
        self.rewind = opt['rewind']
        self.rewind_model_dir = 'rewind_model'

    def train_on_batch(self, label1, label2, data, domain, retrain=True):
        self.network.train()
        self.network.zero_grad()
        data, label1, label2, domain = data.to(self.device), label1.to(self.device), label2.to(self.device), domain.to(
            self.device)
        logit = self.network(data, domain)
        logloss1 = self.criterion(logit[0], label1)
        logloss2 = self.criterion(logit[1], label2)
        if not retrain:
            regloss1 = torch.mean(torch.sigmoid(self.network.temp * self.network.weights[0]))
            regloss2 = torch.mean(torch.sigmoid(self.network.temp * self.network.weights[1]))
            loss = logloss1 + logloss2 + self.reg_lambda1 * regloss1 + self.reg_lambda2 * regloss2
        else:
            loss = logloss1 + logloss2
        loss.backward()
        for optim in self.optim:
            optim.step()
        return loss.item()

    def eval_on_batch(self, data, domain):
        self.network.eval()
        with torch.no_grad():
            data, domain = data.to(self.device), domain.to(self.device)
            logit = self.network(data, domain)
            prob1 = torch.sigmoid(logit[0]).detach().cpu().numpy()
            prob2 = torch.sigmoid(logit[1]).detach().cpu().numpy()
        return prob1, prob2

    def search(self, epochs):
        self.logger.info("ticket:{t}".format(t=self.network.ticket))
        self.logger.info("-----------------Begin Search-----------------")
        cur_auc = 0.0
        for epoch_idx in range(int(epochs)):
            train_loss = .0
            step = 0
            if self.rewind == epoch_idx:
                torch.save(self.network.state_dict(), self.rewind_model_dir)
            if epoch_idx > 0:
                self.network.temp *= self.temp_increase
            for feature, label1, label2, domain in self.dataloader.get_data("train", batch_size=self.bs):
                loss = self.train_on_batch(label1, label2, feature, domain, False)
                train_loss += loss
                step += 1
                if step % 100 == 0:
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                     format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step
            print(torch.sigmoid(self.network.temp * self.network.weights[0]).detach().cpu().numpy())
            print(torch.sigmoid(self.network.temp * self.network.weights[1]).detach().cpu().numpy())
            val_auc_dict, val_loss_dict = self.evaluate_test("validation")
            val_auc, val_loss = 0, 0
            for d in self.domain_list:
                val_auc += (val_auc_dict[d][1] + val_auc_dict[d][2])
                val_loss += (val_loss_dict[d][1] + val_loss_dict[d][2])
            val_auc = val_auc / 6
            val_loss = val_loss / 6
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss:{loss:.6f} | Val AUC:{val_auc:.6f}, Val Loss:{val_loss:.6f}".
                    format(epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))     
                    
        te_auc_dict, te_loss_dict = self.evaluate_test("test")
        for d in self.domain_list:
            self.logger.info(
                "Final Domain {d:d}| Test T1_AUC: {te_auc_t1:.6f}, Test T1_Loss:{te_loss_t1:.6f}, Test T2_AUC: {te_auc_t2:.6f}, Test T2_Loss:{te_loss_t2:.6f}".
                format(epoch=epoch_idx, d=d, te_auc_t1=te_auc_dict[d][1], te_loss_t1=te_loss_dict[d][1],
                       te_auc_t2=te_auc_dict[d][2], te_loss_t2=te_loss_dict[d][2]))

        torch.save({
            'gate_hypernet_state_dict': self.network.gate_hypernet.state_dict(),
            'gate_embedding_state_dict': self.network.embedding.state_dict(),
            'gate_head0_state_dict': self.network.gate_head0.state_dict(),
            'gate_head1_state_dict': self.network.gate_head1.state_dict()
        }, 'gate_params.pth')

    def train(self, epochs):
        self.network = autoifs.AutoIFS(self.opt["model_opt"]).to(self.device)
        self.network.load_state_dict(torch.load(self.rewind_model_dir))
        checkpoint = torch.load('gate_params.pth')
        self.network.gate_hypernet.load_state_dict(checkpoint['gate_hypernet_state_dict'])
        self.network.gate_embedding.load_state_dict(checkpoint['gate_embedding_state_dict'])
        self.network.gate_head0.load_state_dict(checkpoint['gate_head0_state_dict'])
        self.network.gate_head1.load_state_dict(checkpoint['gate_head1_state_dict'])
        self.network.ticket = True

        cur_auc = 0.0
        early_stop = False
        self.optim = autoifs.getOptim(self.network, self.opt["optimizer"], self.r_lr, self.r_l2)[:1]
        self.logger.info("-----------------Begin Train-----------------")
        self.logger.info("Ticket:{t}".format(t=self.network.ticket))
        # ds = self.dataloader.get_train_data("train", batch_size=self.bs)
        for epoch_idx in range(int(epochs)):
            train_loss = .0
            step = 0
            for feature, label1, label2, domain in self.dataloader.get_data("train", batch_size=self.bs):
                loss = self.train_on_batch(label1, label2, feature, domain)
                train_loss += loss
                step += 1
                if step % 100 == 0:
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                     format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step

            val_auc_dict, val_loss_dict = self.evaluate_test("validation")
            val_auc, val_loss = 0, 0
            for d in self.domain_list:
                val_auc += (val_auc_dict[d][1] + val_auc_dict[d][2])
                val_loss += (val_loss_dict[d][1] + val_loss_dict[d][2])
            val_auc = val_auc / 6
            val_loss = val_loss / 6
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss:{loss:.6f} | Val AUC:{val_auc:.6f}, Val Loss:{val_loss:.6f}".
                    format(epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))
            if val_auc > cur_auc:
                cur_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
            else:
                self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                te_auc_dict, te_loss_dict = self.evaluate_test("test")
                for d in self.domain_list:
                    self.logger.info(
                        "Early stop at epoch {epoch:d}|Domain {d:d}| Test T1_AUC: {te_auc_t1:.6f}, Test T1_Loss:{te_loss_t1:.6f}, Test T2_AUC: {te_auc_t2:.6f}, Test T2_Loss:{te_loss_t2:.6f}".
                        format(epoch=epoch_idx, d=d, te_auc_t1=te_auc_dict[d][1], te_loss_t1=te_loss_dict[d][1],
                               te_auc_t2=te_auc_dict[d][2], te_loss_t2=te_loss_dict[d][2]))
                break
        if not early_stop:
            te_auc_dict, te_loss_dict = self.evaluate_test("test")
            for d in self.domain_list:
                self.logger.info(
                    "Final Domain {d:d}| Test T1_AUC: {te_auc_t1:.6f}, Test T1_Loss:{te_loss_t1:.6f}, Test T2_AUC: {te_auc_t2:.6f}, Test T2_Loss:{te_loss_t2:.6f}".
                    format(epoch=epoch_idx, d=d, te_auc_t1=te_auc_dict[d][1], te_loss_t1=te_loss_dict[d][1],
                           te_auc_t2=te_auc_dict[d][2], te_loss_t2=te_loss_dict[d][2]))

    def evaluate(self, on: str):
        preds, trues = {1: [], 2: []}, {1: [], 2: []}
        for feature, label1, label2, domain in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred1, pred2 = self.eval_on_batch(feature, domain)
            label1 = label1.detach().cpu().numpy()
            label2 = label2.detach().cpu().numpy()
            preds[1].append(pred1)
            preds[2].append(pred2)
            trues[1].append(label1)
            trues[2].append(label2)
        y_pred1 = np.concatenate(preds[1]).astype("float64")
        y_true1 = np.concatenate(trues[1]).astype("float64")
        y_pred2 = np.concatenate(preds[2]).astype("float64")
        y_true2 = np.concatenate(trues[2]).astype("float64")
        auc1 = metrics.roc_auc_score(y_true1, y_pred1)
        loss1 = metrics.log_loss(y_true1, y_pred1)
        auc2 = metrics.roc_auc_score(y_true2, y_pred2)
        loss2 = metrics.log_loss(y_true2, y_pred2)
        return auc1, loss1, auc2, loss2

    def evaluate_test(self, on: str):
        preds_dict = {}
        trues_dict = {}
        for d in self.domain_list:
            preds_dict[d] = {1: [], 2: []}
            trues_dict[d] = {1: [], 2: []}
        for feature, label1, label2, domain in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred1, pred2 = self.eval_on_batch(feature, domain)
            label1 = label1.detach().cpu().numpy()
            label2 = label2.detach().cpu().numpy()
            domain = domain.detach().cpu().numpy()
            for d in self.domain_list:
                ind = np.nonzero(domain == d)[0]
                preds_dict[d][1].append(pred1[ind])
                trues_dict[d][1].append(label1[ind])
                preds_dict[d][2].append(pred2[ind])
                trues_dict[d][2].append(label2[ind])
        auc_dict = {}
        loss_dict = {}
        for d in self.domain_list:
            auc_dict[d] = {}
            loss_dict[d] = {}
            y_pred1 = np.concatenate(preds_dict[d][1]).astype("float64")
            y_true1 = np.concatenate(trues_dict[d][1]).astype("float64")
            y_pred2 = np.concatenate(preds_dict[d][2]).astype("float64")
            y_true2 = np.concatenate(trues_dict[d][2]).astype("float64")
            auc_dict[d][1] = metrics.roc_auc_score(y_true1, y_pred1)
            loss_dict[d][1] = metrics.log_loss(y_true1, y_pred1)
            auc_dict[d][2] = metrics.roc_auc_score(y_true2, y_pred2)
            loss_dict[d][2] = metrics.log_loss(y_true2, y_pred2)
        return auc_dict, loss_dict


def main():
    sys.path.extend(["./modules", "./dataloader", "./utils"])
    if args.dataset.lower() == "ml-1m":
        field_dim = trainUtils.get_stats("data/ml-1m/stats_0")
        data_dir = "data/ml-1m/threshold_0"
        field = len(field_dim)
        feature = sum(field_dim)
    elif args.dataset.lower() == "kuairand-pure":
        field_dim = trainUtils.get_stats("data/kuairand-pure/stats_2")
        data_dir = "data/kuairand-pure/threshold_2"
        field = len(field_dim)
        feature = sum(field_dim)
    else:
        print('error dataset')
    model_opt = {
        "latent_dim": args.dim, "feat_num": feature, "field_num": field,
        "mlp_dropout": args.mlp_dropout, "use_bn": args.mlp_bn, "mlp_dims": args.mlp_dims,
        "cross": args.cross, "domain_list": args.domain_list, "task_num": args.task_num, "lora_r": args.lora_r
    }

    opt = {"model_opt": model_opt, "dataset": args.dataset, "model": args.model, "lr": args.lr, "l2": args.l2, "rewind": args.rewind, "r_lr": args.r_lr, "r_l2": args.r_l2,
           "bsize": args.bsize, "epoch": args.max_epoch, "search_epoch": args.search_epoch, "optimizer": args.optim, "data_dir": data_dir,
           "save_dir": args.save_dir, "cuda": args.cuda, "final_temp": args.final_temp, "reg_lambda1": args.reg_lambda1, "reg_lambda2": args.reg_lambda2
           }
    print(opt)
    trainer = Trainer(opt)
    trainer.search(args.search_epoch)
    trainer.train(args.max_epoch)


if __name__ == "__main__":
    """
    python autoifs_trainer.py --dataset 'ml-1m'  
    """
    main()
