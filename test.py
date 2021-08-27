from dataset import XRayDataset
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import f1_score
from datetime import datetime

root = os.getcwd()

test_set = XRayDataset("/data/test/")

net = torch.load(root + '/trained_models/net_baseline')
# unet = torch.load(root + '/trained_models/net_unet')



# Parameters

batch_size = 32
loss_function = nn.NLLLoss()

test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=True,
)

def Test(network, loader):
    # Test the model
    network.eval()

    test_loss = 0
    fscore = []
    corrects = 0
    total_images = 0

    for images, labels in loader:

        with torch.no_grad():

            # images_unet = preprocess(images)
            output = network(images)
            labels = torch.as_tensor(labels)

            test_loss += loss_function(output, labels).item()
            label_pred = torch.max(output, 1)[1]
            corrects += torch.sum(label_pred == labels)
            # for i in range(len(label_pred)):
            #     if label_pred[i] == labels[i]:
            #         correct += 1
            Fscore = f1_score(labels, label_pred, average='macro')
            fscore.append(Fscore)
            total_images += len(output)
        pass

    test_fscore = np.average(np.array(fscore))
    print('corrects/total: {}/{}'.format(corrects, total_images))
    accuracy = int(corrects)/int(total_images)
    print('Accuracy: %f' % accuracy)
    print('Test loss: %.2f' % test_loss)
    print('Test fscore: %f' % test_fscore)
    pass

start_time = datetime.now()

print("Testing baseline...")
Test(net, test_dataloader)


finish_time = datetime.now()
time = finish_time - start_time
print("Time required to test: ", time)