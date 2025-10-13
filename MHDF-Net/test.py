import os
from torch.autograd import Variable
import torch.utils.data
from torch.nn import DataParallel
from  loader import  get_loader
from Mymodel import mymodel
from utils import init_log, progress_bar
import config
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
trainloader, testloader = get_loader()
# read dataset

# define model
net = mymodel()
ckpt = torch.load(config.resume)
net.load_state_dict(state_dict=ckpt['net_state_dict'])
net = net.cuda()
net = DataParallel(net)
creterion = torch.nn.CrossEntropyLoss()

# evaluate on train set
train_loss = 0
train_correct = 0
total = 0
net.eval()


# evaluate on test set
test_loss = 0
test_correct = 0
total = 0
for i, data in enumerate(testloader):
    with torch.no_grad():
        img, label, metadata_batch = data[0].cuda(), data[1].cuda(), data[2].cuda()
        metadata_batch = metadata_batch.float()
        batch_size = img.size(0)
        pred = net(img, metadata_batch)
        # calculate loss
        concat_loss = creterion(pred, label)
        # calculate accuracy
        _, concat_predict = torch.max(pred, 1)
        total += batch_size
        test_correct += torch.sum(concat_predict.data == label.data)
        test_loss += concat_loss.item() * batch_size
        progress_bar(i, len(testloader), 'eval on test set')

test_acc = float(test_correct) / total
test_loss = test_loss / total
print('test set loss: {:.3f} and test set acc: {:.3f} total sample: {}'.format(test_loss, test_acc, total))

print('finishing testing')
