import torch
import torchvision
import torch.optim as optim
from tqdm.auto import tqdm
from torch.nn import functional as F
import numpy as np
import datetime
from pytz import timezone
import os
import matplotlib.pyplot as plt

from dataset import CustomDataset, CustomTransform
from model import CustomModel

class Trainer:
    @property
    def USE_CUDA(self) -> bool:
        return torch.cuda.is_available()
        
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device if self.USE_CUDA else "cpu")
        print("사용하는 Device : ", self.device)

        self.data_root = self.cfg.data.data_root
        self.train_dir = self.cfg.data.train_dir
        self.test_dir = self.cfg.data.test_dir
        self.img_resize = self.cfg.data.img_resize
        self.img_resize_center = self.cfg.data.img_resize_center
        self.color_depth = self.cfg.data.color_depth

        self.epochs = self.cfg.train.epochs
        self.batch_size = self.cfg.train.batch_size
        self.learning_rate = self.cfg.train.learning_rate

        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.loss_function = None
        self.optimizer = optim.Adam if self.cfg.train.optim == 'adam' else None

    def train_one_epoch(self, epoch, model, dataloader, optimizer, msg=None):
        model.train()
        train_loss = 0
        for batch_idx, (data, label, _) in enumerate(dataloader):
            data = data.to(self.device)
            optimizer.zero_grad()

            gt = label
            out = model(data)

            loss = self.loss_function(gt, out)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()
                
        print("{}======> Epoch: {} Average loss: {:.4f}".format(
            msg if msg else '=', epoch, train_loss / len(dataloader.dataset)
        ))        
        return train_loss / len(dataloader.dataset)

    def test_one_epoch(self, epoch, model, dataloader, msg=None):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (data, label, _) in enumerate(dataloader):
                data = data.to(self.device)
                
                gt = label
                out = model(data)
                #recon_batch = torch.reshape(recon_batch, (-1, COLOR_DEPTH, IMG_RESIZE, IMG_RESIZE))
                loss = self.loss_function(gt, out)
                
                test_loss += loss.item()

        print("{}======> Epoch: {} Average loss: {:.4f}".format(
            msg if msg else '=', epoch, test_loss / len(dataloader.dataset)
        ))   
        return test_loss / len(dataloader.dataset)

    def train(self):
        exp_num = 1
        total_exp = len(self.epochs)*len(self.batch_size)
        tf, tf_inv = CustomTransform(self.img_resize, self.img_resize_center).get()
        for e in self.epochs:
            for b in self.batch_size:
                current_time = datetime.datetime.now(timezone('Asia/Seoul'))
                current_time = current_time.strftime('%Y-%m-%d-%H:%M')

                saved_loc = os.path.join('./outs', current_time)
                if not os.path.exists(saved_loc):
                    os.mkdir(saved_loc)

                print("저장 위치: ", saved_loc)
                    
                # Loading trainset, testset and trainloader, testloader
                trainset = CustomDataset(root = self.data_root, train = True, transform = tf, train_dir=self.train_dir, test_dir=self.test_dir)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size = b, shuffle = True, num_workers = 2)

                testset = CustomDataset(root = self.data_root, train = False, transform = tf, train_dir=self.train_dir, test_dir=self.test_dir)
                testloader = torch.utils.data.DataLoader(testset, batch_size = b, shuffle = True, num_workers = 2)

                # Model and training settings
                model = CustomModel(self.img_resize*self.img_resize*self.color_depth, 512, 256, 128).to(self.device)
                lr = self.learning_rate
                optimizer = self.optimizer(model.parameters(), lr = self.learning_rate)

                train_losses = []
                test_losses = []
                best_loss = float('inf')
                best_model = None
                # train/test
                for epoch in tqdm(range(0, e)):
                    train_loss = self.train_one_epoch(epoch, model, trainloader, optimizer, msg=f'Experiment # {exp_num} / {total_exp} : saved at {saved_loc}')
                    test_loss = self.test_one_epoch(epoch, model, testloader)
                    
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    if test_loss < best_loss:
                        print("Best Epoch!!", epoch)                    
                        best_loss = test_loss
                        best_model = model
                        # save model
                        torch.save({
                            'config':self.cfg.cfg_path,
                            'model_state_dict':best_model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'train_loss': train_loss,
                            'test_loss' : test_loss,
                            'epoch':epoch
                            }, os.path.join(saved_loc, 'model.pt'))
                
                plt.plot(train_losses)
                plt.plot(test_losses)
                plt.title(saved_loc)
                plt.savefig(os.path.join(saved_loc, 'results.png'), dpi=300)
                exp_num += 1

def inference():
    pass
def evaluate():
    pass

if __name__=="__main__":
    from configs.config import Config
    cfg = Config('configs/sample.yaml')
    trainer = Trainer(cfg)
    print(trainer.data_root)