import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
from Final_contest.config import Config
from Final_contest.dataloader import DataSet
from Final_contest.network import Network, My_mse_loss
from Final_contest.evaluation import calculateAllMetrics
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import os
opt = Config()

# prepare dataset
print("==> loading data...")
data = DataSet(opt)
trainset = data.get_Trainset()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.TRAIN_BATCH_SIZE, shuffle=True)

valset = data.get_Valset()
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.VAL_BATCH_SIZE, shuffle=True)

print("==> load data successfully")

print("==> get weight of cross entropyloss")
weight_hour = np.zeros(shape=[24], dtype=np.float32)

for i in tqdm(range(len(trainset))):
    weight_hour[trainset[i][2]] += 1

    # if temp > 0:
    #     weight_day[temp - 1] += 1
    # else:
    #     weight_day[temp] += 1
print(weight_hour)

hour_log = np.log(weight_hour)
weight_hour = (-(hour_log - hour_log.max()) + 0.5) / 0.5

net = Network(opt).cuda()
print("==> net infomation:")
print(net)
# load model
if os.path.exists(opt.MODEL_SAVE_PATH) and opt.USING_MODEL:
    net = torch.load(opt.MODEL_SAVE_PATH)
    print("==> load model successfully")
else:
    print("==> model file dose not exist : ", opt.MODEL_SAVE_PATH)
# setup network
if opt.USE_CUDA:
    print("==> using CUDA")
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count())).cuda()
    cudnn.benchmark = True

# set criterion (loss function)
criterion_day = My_mse_loss()
criterion_hour = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weight_hour).cuda())

# you can choose metric in [accuracy, MSE, RankScore]
highest_metrics = 60
lower_percent = 0.98


def train(epoch):
    net.train()
    print("train epoch:", epoch)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.get_lr(epoch), weight_decay=1e-3)
    loss_day = 0
    loss_hour = 0
    for batch_idx, (inputs, targets_sign_day, targets_sign_hour) in enumerate(tqdm(trainloader)):
        if opt.USE_CUDA:
            inputs = inputs.cuda()
            targets_sign_day = targets_sign_day.cuda()
            targets_sign_hour = targets_sign_hour.cuda()

        (output_hour, output_day) = net(inputs.float())

        output_day = output_day.view([-1, opt.OUTPUT_DAY_SIZE])
        output_hour = output_hour.view([-1, opt.OUTPUT_HOUR_SIZE])

        loss_day = criterion_day(output_day, targets_sign_day)
        loss_hour = criterion_hour(output_hour, targets_sign_hour)
        loss = loss_day + loss_hour

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        print("==> epoch {}: loss_day is {}, loss_hour is {} ".format(epoch, loss_day, loss_hour))

def val(epoch):
    global highest_metrics
    net.eval()
    pred_signed_time = []
    real_signed_time = []
    for batch_idx, (inputs,payed_time, signed_time) in enumerate(tqdm(valloader)):
        if opt.USE_CUDA:
            inputs = inputs.cuda()

        (output_hour, output_day) = net(inputs.float())

        output_hour = output_hour.data.cpu().numpy()
        # transform output_day into integer
        output_day = np.floor(output_day.data.cpu().numpy() + 1 - opt.Threshold)
        # calculate pred_signed_time via output
        for i in range(len(inputs)):

            pred_time_hour = np.argmax(output_hour[i])
            pred_time_day = output_day[i]
            temp_payed_time = payed_time[i]
            temp_payed_time = datetime.datetime.strptime(temp_payed_time, "%Y-%m-%d %H:%M:%S")

            temp_pred_signed_time = temp_payed_time.replace(hour=int(pred_time_hour)) + relativedelta(days=int(pred_time_day))
            temp_pred_signed_time = temp_pred_signed_time.replace(hour=int(pred_time_hour))
            temp_pred_signed_time = temp_pred_signed_time.replace(minute=0)
            temp_pred_signed_time = temp_pred_signed_time.replace(second=0)

            pred_signed_time.append(temp_pred_signed_time.strftime("%Y-%m-%d %H"))
            real_signed_time.append(signed_time[i])

    (rankScore_result, onTimePercent_result, accuracy_result) = calculateAllMetrics(real_signed_time, pred_signed_time)
    print("==> epoch {}: rankScore is {}, onTimePercent is {}, accuracy is {}".format(epoch, rankScore_result,
                                                                                      onTimePercent_result,
                                                                                      accuracy_result))

    # save model
    if rankScore_result < highest_metrics and onTimePercent_result > lower_percent:
        print("==> saving model")
        print("==> onTimePercent {} | rankScore {} ".format(onTimePercent_result, rankScore_result))
        highest_metrics = rankScore_result
        torch.save(net, opt.MODEL_SAVE_PATH )

# start training
if __name__ == '__main__':
    for i in range(opt.NUM_EPOCHS):
        train(i)
        if i % opt.val_step == 0:
            val(i)
