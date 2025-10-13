import os
import pandas as pd
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from utils import init_log, progress_bar
from torch.autograd import Variable
from loader import get_loader
from  Mymodel import mymodel_MetaNet,mymodel_Cat,mymodel_MetaBlock,mymodel_ours,mymodel_MLDF
import numpy as np
import random
import config
#from resnet import resnet50

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

setup_seed(3407)

resume = config.resume

SAVE_FREQ = 10
save_dir = config.model_dir
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
start_epoch = 1

save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))

if os.path.exists(save_dir):
    raise NameError('model dir exists!')

os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda")

train_data_loader, test_data_loader = get_loader()

net = mymodel_Cat()


if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

cross_entropy_loss = torch.nn.CrossEntropyLoss().to(device)

_sched_factor = 0.1
_sched_min_lr = 1e-6
_sched_patience = 10
_early_stop = 15
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                       patience=_sched_patience)
print("mobilenet-concat")

net = net.cuda()
net = DataParallel(net)

acc = []
best_acc = 0

data = pd.DataFrame()

for epoch in range(start_epoch, config.epoch):


    # begin training
    _print('--' * 50)
    net.train()
    for step, (img, label, metadata, img_id) in enumerate(train_data_loader):

        img0 = img.to(device)
        classlabel1 = label.to(device)
        metadata_batch = metadata.cuda()
        metadata_batch = metadata_batch.float()

        batch_size = img.size(0)
        optimizer.zero_grad()

        y_pred_cat = net(img0, metadata_batch)
        batch_loss = cross_entropy_loss(y_pred_cat, classlabel1)
        batch_loss.backward()
        optimizer.step()

        progress_bar(step, len(train_data_loader), 'train')

    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct = 0
        total = 0

        net.eval()
        for step,(img, label, metadata, img_id) in enumerate(train_data_loader):

            with torch.no_grad():
                img0 = img.to(device)
                classlabel1 = label.to(device)
                metadata_batch = metadata.cuda()
                metadata_batch = metadata_batch.float()
                batch_size = img.size(0)
                y_pred_cat = net(img0, metadata_batch)
                batch_loss = cross_entropy_loss(y_pred_cat, classlabel1)
                _, concat_predict = torch.max(y_pred_cat, 1)
                total += batch_size

                train_correct += torch.sum(concat_predict.data == classlabel1.data)

                train_loss += batch_loss.item() * batch_size
                progress_bar(step, len(train_data_loader), 'eval train set')


        train_acc = float(train_correct) / total
        train_loss = train_loss / total

        _print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                total))

	# evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        for step, (img, label1, metadata, img_id) in enumerate(test_data_loader):
            with torch.no_grad():
                img0 = Variable(img).to(device)
                metadata_batch = metadata.cuda()
                metadata_batch = metadata_batch.float()

                classlabel1 = label1.to(device)
                batch_size = img.size(0)
                y_pred_raw= net(img0, metadata_batch)
                # calculate loss
                concat_loss = cross_entropy_loss(y_pred_raw, classlabel1)
                # calculate accuracy
                _, concat_predict = torch.max(y_pred_raw, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == classlabel1.data)
                test_loss += concat_loss.item() * batch_size
                progress_bar(step, len(test_data_loader), 'eval test set')

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))

	# save model
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

print('finishing training')
