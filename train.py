import torch
import torch.nn as nn
from torch.optim import Adam


from dataset import flare_dataloader
from model import define_model
import utils
import time
from tqdm import tqdm



class FlareModel:

    def __init__(self, opt):
        print("==>> Training Flare Removal Model")

        self.opt = opt
        self.initial_lr = opt.learning_rate
        self.n_epoch = opt.n_epochs
        self.b1 = opt.b1
        self.b2 = opt.b2
        self.batch_size = opt.batch_size

        self.load_height = opt.load_height
        self.load_width = opt.load_width

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transforms = utils.get_transforms(self.load_height, self.load_width)

        self.dataset = flare_dataloader(self.opt, self.transforms)

        self.train_loader = self.dataset.train_loader
        self.test_loader = self.dataset.test_loader

        self.model = define_model(3, 3, opt.ngf,'batch', False, 'normal', 0.02, gpu_id = self.device)
        self.optimizer = Adam(self.model.parameters(), lr = self.initial_lr, betas = (self.b1, self.b2))

        self.loss_function = torch.nn.MSELoss().to(self.device)

    def train(self):

        epoch_loss_list = []

        for epoch in range(self.n_epoch):
            running_loss = 0
            counter = 0
            start_time = time.time()
            
            self.model.train()
            train_loop = tqdm(self.train_loader , leave=True , desc=f"Epoch {epoch:>2}")

            for batch_idx, (train_img, train_label) in enumerate(train_loop):
                train_img = train_img.to(self.device)
                train_label = train_label.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(train_img)

                loss = self.loss_function(outputs, label)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * train_img.size(0)

            epoch_loss = running_loss / len(self.train_loader)

            print("[TRAIN] Epoch: {}/{} Train Loss: {}".format(epoch+1, self.n_epoch, epoch_loss))


            epoch_loss_list.append(epoch_loss)

            if epoch_loss == min(epoch_loss_list):
                self.save_model()

            model.eval()

            test_running_loss = 0.0

            test_loop = tqdm(self.test_loader , leave=True , desc=f"Epoch {epoch:>2}")

            for test_batch_idx, (test_img, test_label) in enumerate(test_loop):
                test_img = test_img.to(self.device)
                test_label = test_label.to(self.device)

                with torch.no_grad():
                    test_output = self.model(test_img)
                
                test_loss = self.loss_function(test_output, test_label)

                test_running_loss += test_loss.item() * test_img.size(0)
            
            test_epoch_loss = test_running_loss / len(self.test_loader)

            print("[TEST] Epoch: {}/{} Test Loss: {}".format(epoch+1, self.n_epoch, test_epoch_loss))

            end = time.time()

            print("Time taken for epoch {}: {} min".format(epoch+1, (end-start)/60))
                    


    def save_model(self):
        checkpoint_path = self.opt.checkpoint_dir
        save_path = os.path.join(checkpoint_path, "model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"[+] Saving weights to {save_path}")





        

