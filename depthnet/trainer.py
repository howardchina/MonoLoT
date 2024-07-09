import os
from pathlib import Path
from datetime import datetime
from datetime import datetime
import torch
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter

from .utils import *
from .datasets.gastro_dataset import get_data_loaders
from . import meters


class Trainer():
    def __init__(self, cfgs, model):
        self.cfgs = cfgs
        self.log_path = Path(cfgs.get('log_dir', 'results')) / cfgs.get('model_name', datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = cfgs.get('batch_size', 64)
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.start_epoch = cfgs.get('start_epoch', 0)
        self.load_weights_folder = cfgs.get('load_weights_folder', None)
        self.mypretrain = cfgs.get('mypretrain', None)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.run_val = cfgs.get('run_val', True)
        self.run_test = cfgs.get('run_test', False)
        # geometry loss
        self.geometry_loss = cfgs.get('geometry_loss', False)
        self.supervised = cfgs.get('supervised', False)
        

        self.model = model(cfgs)
        self.model.trainer = self
        # self.train_loader, self.val_loader, self.test_loader, info_dict = get_data_loaders(cfgs)
        data_loaders = get_data_loaders(cfgs)
        self.train_loader = data_loaders["train_loader"]
        if self.run_val:
            self.val_loader = data_loaders["val_loader"]
            self.val_iter = iter(self.val_loader)
        if self.run_test:
            self.test_loader = data_loaders["test_loader"]
        info_dict = data_loaders["info_dict"]
        # self.train_loader, self.val_loader, info_dict = get_data_loaders(cfgs)
        
        
        cfgs.update(info_dict)
        self.num_total_steps = info_dict['num_total_steps']
        if self.start_epoch > 0:
            self.num_total_steps -= self.num_total_steps / self.num_epochs * self.start_epoch
        self.metrics_trace = meters.MetricsTrace()
        self.metric_str_exclude = cfgs.get('metric_str_exclude', [])
        self.make_metrics = lambda m=None: meters.StandardMetrics(m, self.metric_str_exclude)

    def load_checkpoint(self):
        
        if self.mypretrain is not None:
            self.model.load_pretrain()
            
        if self.load_weights_folder is not None:
            self.model.load_model()
            
    def save_checkpoint(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if self.epoch+1 == self.num_epochs:
            last_folder = os.path.join(self.log_path, "models", "weights_last")
            os.symlink("weights_{}".format(self.epoch), last_folder)
            
        for net_name in self.model.network_names:
            save_path = os.path.join(save_folder, "{}.pth".format(net_name))
            to_save = getattr(self.model, net_name).state_dict()
            if net_name == "net_depth_encoder":
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.model.height
                to_save['width'] = self.model.width
            torch.save(to_save, save_path)
        
        for optim_name in self.model.optimizer_names:
            save_path = os.path.join(save_folder, "{}.pth".format(optim_name))
            torch.save(getattr(self.model, optim_name).state_dict(), save_path)

    def train(self):
        """Perform training."""
        dump_yaml(Path(self.log_path) / 'models' / 'configs.yml', self.cfgs)

        ## initialize
        self.metrics_trace.reset()
        self.start_time = time.time()
        self.train_iter_per_epoch = len(self.train_loader)
        if self.run_val:
            self.val_iter_per_epoch = len(self.val_loader)
        if self.run_test:
            self.test_iter_per_epoch = len(self.test_loader)
        self.model.to_device(self.device)
        self.model.init_optimizers()

        ## resume from checkpoint
        self.load_checkpoint()

        ## initialize tensorboardX logger
        self.writers = {}
        
        for mode in ["train", "val", "test"]:
            self.writers[mode] = SummaryWriter(Path(self.log_path) / "logs" / mode)
        
        print("Training model named:\n  ", self.model.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        ## run epochs
        print(f"{self.model.model_name}: optimizing to {self.num_epochs} epochs")
        if self.start_epoch > 0:
            for _ in range(self.start_epoch):
                for scheduler_name in self.model.scheduler_names:
                    getattr(self.model, scheduler_name).step()
        for self.epoch in range(self.start_epoch, self.num_epochs):
            metrics = self.run_epoch(self.train_loader, self.epoch, is_train=True)
            self.metrics_trace.append("train", metrics)
            
            if self.run_val:
                with torch.no_grad():
                    metrics = self.run_epoch(self.val_loader, self.epoch, is_train=False, is_val=True)
                    self.metrics_trace.append("val", metrics)
                    
            if self.run_test:
                with torch.no_grad():
                    metrics = self.run_epoch(self.test_loader, self.epoch, is_train=False, is_test=True)
                    self.metrics_trace.append("test", metrics)

            if (self.epoch+1) % self.save_checkpoint_freq == 0 or (self.epoch + 1) == self.num_epochs:
                self.save_checkpoint()
            self.metrics_trace.plot(pdf_path=os.path.join(self.log_path, 'metrics.pdf'))
            self.metrics_trace.save(os.path.join(self.log_path, 'metrics.json'))

        print(f"Training completed after {self.epoch+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_train=True, is_val=False, is_test=False):
        """Run one epoch."""
        metrics = self.make_metrics()
        
        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
            mode = "train"
            iter_per_epoch = self.train_iter_per_epoch
            for scheduler_name in self.model.scheduler_names:
                getattr(self.model, scheduler_name).step()
        elif is_val:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()
            mode = "val"
            iter_per_epoch = self.val_iter_per_epoch
        elif is_test:
            print(f"Starting testing epoch {epoch}")
            self.model.set_eval()
            mode = "test"
            iter_per_epoch = self.test_iter_per_epoch
        else:
            raise NotImplementedError
            

        for batch_idx, inputs in enumerate(loader):
            before_op_time = time.time()
            
            outputs, losses = self.model.forward(inputs)
            
            if is_train:
                self.model.backward(losses)
                
            duration = time.time() - before_op_time
            metrics.update(losses, self.batch_size)
            
            # visualization
            if self.use_logger and is_train:
                total_iter = batch_idx + epoch * self.train_iter_per_epoch
                if total_iter % self.log_freq == 0:
                    iter_sofar = batch_idx + (epoch - self.start_epoch) * self.train_iter_per_epoch
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data, iter_sofar)

                    if "depth_gt" in inputs:
                        depth_losses = self.model.compute_depth_losses(inputs, outputs)
                        metrics.update(depth_losses, self.batch_size)
                        losses.update(depth_losses)
                    self.log('train', inputs, outputs, losses, total_iter)
                    
            elif self.use_logger and is_val:
                total_iter = batch_idx + epoch * self.val_iter_per_epoch
                if "depth_gt" in inputs:
                    depth_losses = self.model.compute_depth_losses(inputs, outputs)
                    metrics.update(depth_losses, self.batch_size)
                    losses.update(depth_losses)
                if total_iter % self.log_freq == 0:
                    self.log('val', inputs, outputs, losses, total_iter)
            
            elif self.use_logger and is_test:
                total_iter = batch_idx + epoch * self.test_iter_per_epoch
                if "depth_gt" in inputs:
                    depth_losses = self.model.compute_depth_losses(inputs, outputs)
                    metrics.update(depth_losses, self.batch_size)
                    losses.update(depth_losses)
                if total_iter % self.log_freq == 0:
                    self.log('test', inputs, outputs, losses, total_iter)
            
            # log on terminal
            if batch_idx < 10 or batch_idx % 10 == 0:
                print(f"{mode}{epoch:02}/{batch_idx:05}/{iter_per_epoch:05}/{metrics}")
        
        # after run_epoch
        if self.use_logger:
            for k, v in metrics.get_data_dict().items():
                self.writers[mode].add_scalar(f'Metrics/{k}', v, epoch)
        print("{}/Metrics:".format(mode))
        print(metrics.get_data_dict())
        
        return metrics

    def log(self, mode, inputs, outputs, losses, step):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, step)

        for j in range(min(4, self.batch_size)):  # write a maxmimum of four images
            for s in self.model.scales:
                for frame_id in self.model.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, step)

                writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), step)

                if not self.model.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], step)
                    
                if self.geometry_loss:
                    for frame_id in self.model.frame_ids[1:]:
                        writer.add_image(
                                "computed_depth_{}_{}/{}".format(frame_id, s, j),
                                normalize_image(outputs[("computed_depth", frame_id, s)][j]), step)
                        writer.add_image(
                                "sampled_depth_{}_{}/{}".format(frame_id, s, j),
                                normalize_image(outputs[("sampled_depth", frame_id, s)][j]), step)
                    
    def log_time(self, batch_idx, duration, loss, step):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / step - 1.0) * time_sofar if step > 0 else 0
        if self.supervised:
            print_string = "epoch {:>3} | lr {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
            print(print_string.format(self.epoch, self.model.optimizer_depth.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        else:
            print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
            print(print_string.format(self.epoch, self.model.optimizer_depth.state_dict()['param_groups'][0]['lr'],
                                  self.model.optimizer_pose.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))