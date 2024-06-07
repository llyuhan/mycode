import argparse

from tensorboardX import SummaryWriter

import random

import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.vnet_sdf import VNet
from utils import ramps, losses
from dataloaders.la_heart import *

class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_data_path = args.root_path
        self.snapshot_path = "../model/" + args.exp + "/"
        self.num_classes = 2
        self.patch_size = (112, 112, 80)
        self.weights = np.ones(args.size_unlabeled, dtype=np.float32)
        self.m_ts = np.zeros(args.size_unlabeled, dtype=np.float32)
        self.v_ts = np.zeros(args.size_unlabeled, dtype=np.float32)
        self.ts = np.zeros(args.size_unlabeled, dtype=np.float32)
        self.alpha = args.alpha
        self.inf_warm = args.inf_warm
        self.inner_steps = args.inner_steps

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.batch_size = args.batch_size * len(args.gpu.split(','))
        self.max_iterations = args.max_iterations
        self.base_lr = args.base_lr
        self.labeled_bs = args.labeled_bs

        if args.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

        self.model = self.create_model()
        self.ema_model = self.create_model(ema=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=0.0001)

        if args.consistency_type == 'mse':
            self.consistency_criterion = losses.softmax_mse_loss
        elif args.consistency_type == 'kl':
            self.consistency_criterion = losses.softmax_kl_loss
        else:
            assert False, args.consistency_type

        self.writer = SummaryWriter(self.snapshot_path + '/log')

    def create_model(self, ema=False):
        net = VNet(n_channels=1, n_classes=self.num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(self, worker_id):
        random.seed(self.args.seed + worker_id)

    def cal_dice(self, output, target, eps=1e-3):
        output = torch.argmax(output, dim=1)
        inter = torch.sum(output * target) + eps
        union = torch.sum(output) + torch.sum(target) + eps * 2
        dice = 2 * inter / union
        return dice

    def get_current_consistency_weight(self, epoch):
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))

    def compute_influence(self, grad_vec_wrt_val_loss, x, y):
        inv_H = self.compute_inverse_hessian(x, y)
        grads_train_per_ex = self.compute_grads_train_per_ex(x, y)
        jacobian_per_ex_u = grads_train_per_ex[self.labeled_bs:]
        jacobian_per_ex_u = jacobian_per_ex_u.reshape(self.labeled_bs, -1).transpose()
        grad_vec_wrt_val = grad_vec_wrt_val_loss.reshape(1, -1)
        influences = -np.matmul(np.matmul(grad_vec_wrt_val, inv_H), jacobian_per_ex_u)[0]
        return influences

    def compute_inverse_hessian(self, x, y):
        self.model.train()
        hessian = None
        for i in range(x.size(0)):
            self.optimizer.zero_grad()
            output = self.model(x[i].unsqueeze(0))
            loss = F.cross_entropy(output, y[i].unsqueeze(0))
            loss.backward(create_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.append(param.grad.view(-1))
            grad = torch.cat(grad)
            hessian_i = torch.autograd.grad(grad, self.model.parameters(), grad_outputs=grad, retain_graph=True)
            hessian_i = torch.cat([h.view(-1) for h in hessian_i])
            if hessian is None:
                hessian = hessian_i
            else:
                hessian += hessian_i
        hessian /= x.size(0)
        inv_hessian = torch.inverse(hessian.view(-1, 1))
        return inv_hessian

    def compute_grads_train_per_ex(self, x, y):
        self.model.train()
        grads = []
        for i in range(x.size(0)):
            self.optimizer.zero_grad()
            output = self.model(x[i].unsqueeze(0))
            loss = F.cross_entropy(output, y[i].unsqueeze(0))
            loss.backward()
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.append(param.grad.view(-1))
            grads.append(torch.cat(grad))
        grads = torch.stack(grads)
        return grads

    def grad_wrt_val_loss_batch(self):
        self.model.eval()
        val_data = LAHeart(base_dir=self.train_data_path, split='val', transform=transforms.Compose([ToTensor()]))
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

        val_loss = 0
        for i_batch, sampled_batch in enumerate(val_loader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = self.model(volume_batch)
            loss = F.cross_entropy(outputs, label_batch)
            val_loss += loss.item()

        val_loss /= len(val_loader)

        self.model.train()
        return val_loss

    def identify_and_correct_labels(self, train_loader):
        self.model.eval()
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = self.model(volume_batch)
            predicted_labels = torch.argmax(outputs, dim=1)
            # 假设影响函数值较大的样本是错误标签
            influences = self.compute_influence(self.grad_wrt_val_loss_batch(), volume_batch, label_batch)
            error_indices = influences > np.percentile(influences, 95)  # 取前5%的样本作为错误标签
            label_batch[error_indices] = predicted_labels[error_indices]
            # 更新训练数据集中的标签
            sampled_batch['label'] = label_batch.cpu()

    def train_step(self, train_loader, test_loader, iter_num, ep):
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[self.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            outputs = self.model(volume_batch)
            with torch.no_grad():
                ema_output = self.ema_model(ema_inputs)

            loss_seg = F.cross_entropy(outputs[:self.labeled_bs], label_batch[:self.labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:self.labeled_bs, 1, :, :, :], label_batch[:self.labeled_bs] == 1)
            supervised_loss = 0.5 * (loss_seg + loss_seg_dice)

            consistency_weight = self.get_current_consistency_weight(iter_num // 150)
            consistency_dist = self.consistency_criterion(outputs[self.labeled_bs:], ema_output)
            consistency_loss = consistency_weight * consistency_dist
            loss = supervised_loss + consistency_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_ema_variables(self.model, self.ema_model, self.args.ema_decay, iter_num)

            if ep > self.inf_warm and iter_num % (self.inner_steps * self.args.batch_size) == 0:
                grad_vec_wrt_val_loss = self.grad_wrt_val_loss_batch()
                influences = self.compute_influence(grad_vec_wrt_val_loss, volume_batch, label_batch)
                batch_ids = sampled_batch['index'][:, 0]
                self.identify_and_correct_labels(train_loader)

            if iter_num % 20 == 0:
                self.writer.add_scalar('train/loss', loss, iter_num)
                self.writer.add_scalar('train/supervised_loss', supervised_loss, iter_num)
                self.writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                self.writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)

            if iter_num % 200 == 0:
                self.model.eval()
                dice_sample = []
                for i_batch, sampled_batch in enumerate(test_loader):
                    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    outputs = self.model(volume_batch)
                    outputs_soft = F.softmax(outputs, dim=1)
                    dice = self.cal_dice(outputs_soft[:, 1, :, :, :], label_batch == 1)
                    dice_sample.append(dice.item())
                mean_dice = np.mean(dice_sample)
                self.writer.add_scalar('test/dice', mean_dice, iter_num)
                self.model.train()

            iter_num += 1
            if iter_num >= self.max_iterations:
                break

    def train(self):
        train_data = LAHeart(base_dir=self.train_data_path, split='train', transform=transforms.Compose([
            RandomRotFlip(), RandomCrop(self.patch_size), ToTensor()]))
        labeled_idxs = list(range(0, 16))
        unlabeled_idxs = list(range(16, 80))
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, self.batch_size,
                                              self.batch_size - self.labeled_bs)
        train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                                  worker_init_fn=self.worker_init_fn)

        test_data = LAHeart(base_dir=self.train_data_path, split='test', transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

        iter_num = 0
        max_epoch = self.max_iterations // len(train_loader) + 1
        for epoch_num in (tqdm(range(max_epoch), ncols=70)):
            self.train_step(train_loader, test_loader, iter_num, epoch_num)
            if iter_num >= self.max_iterations:
                break

        self.writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../data/LA/2018LA_Seg_TrainingSet',
                        help='Name of Experiment')
    parser.add_argument('--exp', type=str, default='LA_VNet', help='experiment_name')
    parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--size_unlabeled', type=int, default=64, help='size of unlabeled data')
    parser.add_argument('--alpha', type=float, default=0.01, help='learning rate to update lambda')
    parser.add_argument('--inf_warm', type=int, default=0, help='influence computing warm-up')
    parser.add_argument('--inner_steps', type=int, default=100, help='how often to update lambdas')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()