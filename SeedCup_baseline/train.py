import torch
import os
from SeedCup_baseline.config import Config
from SeedCup_baseline.model import My_MSE_loss
import torch.backends.cudnn as cudnn
from SeedCup_baseline.dataLoader import TrainSet, ValSet
from SeedCup_baseline.model import Network
from SeedCup_baseline.evaluation import calculateAllMetrics
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

opt = Config()

# prepare dataset
print("==> loading data...")
trainset = TrainSet(opt.TRAIN_FILE, opt=opt)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.TRAIN_BATCH_SIZE, shuffle=True)

valset = ValSet(opt.VAL_FILE, opt=opt)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.VAL_BATCH_SIZE, shuffle=False)
print("==> load data successfully")
# load model
# if os.path.exists(opt.MODEL_SAVE_PATH):
#     net = torch.load(opt.MODEL_SAVE_PATH)
#     print("==> load model successfully")
# else:
#     print("==> model file dose not exist : ", opt.MODEL_SAVE_PATH)
# setup network
net = Network(opt)
if opt.USE_CUDA:
    print("==> using CUDA")
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count())).cuda()
    cudnn.benchmark = True
# set criterion (loss function)
criterion_1 = torch.nn.MSELoss()
criterion_2 = My_MSE_loss()

# you can choose metric in [accuracy, MSE, RankScore]
highest_metrics = 100
lower_percent = 0.975


def train(epoch):
    net.train()
    print("train epoch:", epoch)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.get_lr(epoch))
    for batch_idx, (inputs, targets_sign_day, targets_sign_hour) in enumerate(tqdm(trainloader)):
        if opt.USE_CUDA:
            inputs = inputs.cuda()
            targets_sign_day = targets_sign_day.cuda()
            targets_sign_hour = targets_sign_hour.cuda()

        inputs = torch.autograd.Variable(inputs)
        targets_sign_day = torch.autograd.Variable(targets_sign_day.float())
        targets_sign_hour = torch.autograd.Variable(targets_sign_hour.float())

        optimizer.zero_grad()

        (output_FC_1_1, output_FC_1_2) = net(inputs.float())

        output_FC_1_1 = output_FC_1_1.reshape(-1)

        output_FC_1_2 = output_FC_1_2.reshape(-1)

        loss_1_1 = criterion_2(output_FC_1_1, targets_sign_day)

        loss_1_2 = criterion_1(output_FC_1_2, targets_sign_hour)

        loss_day = loss_1_1
        loss_hour = loss_1_2

        loss = loss_day + loss_hour
        loss.backward()

        optimizer.step()

        # TODO add to tensorboard
    print("==> epoch {}: loss_day is {}, loss_hour is {} ".format(epoch, loss_day, loss_hour))


def val(epoch):
    global highest_metrics
    net.eval()
    pred_signed_time = []
    real_signed_time = []
    for batch_idx, (inputs, payed_time, signed_time) in enumerate(tqdm(valloader)):
        if opt.USE_CUDA:
            inputs = inputs.cuda()

        inputs = torch.autograd.Variable(inputs)
        (output_FC_1_1, output_FC_1_2) = net(inputs.float())

        # calculate pred_signed_time via output
        for i in range(len(inputs)):
            pred_time_day = output_FC_1_1[i]
            pred_time_hour = output_FC_1_2[i]
            # if output_FC_1_2[i] < 10:
            #     pred_time_hour = 10
            # elif output_FC_1_2[i] > 19:
            #     pred_time_hour = 19
            temp_payed_time = payed_time[i]
            temp_payed_time = datetime.datetime.strptime(temp_payed_time, "%Y-%m-%d %H:%M:%S")
            temp_payed_time = temp_payed_time.replace(hour=int(pred_time_hour))

            temp_pred_signed_time = temp_payed_time + relativedelta(days=int(pred_time_day))
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
        torch.save(net, opt.MODEL_SAVE_PATH)


# start training
if __name__ == '__main__':
    for i in range(opt.NUM_EPOCHS):
        train(i)

        if i % opt.val_step == 0:
            val(i)
