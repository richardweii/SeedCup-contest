import torch
from Final_contest.config import Config
from Final_contest.dataloader import TestSet
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import numpy as np
import os

'''
do inference on TestSet and output txt file
'''
def get_test_result():
    opt = Config()

    # load model
    if os.path.exists(opt.MODEL_SAVE_PATH):
        net = torch.load(opt.MODEL_SAVE_PATH)
        print("==> load model successfully")
    else:
        print("==> model file dose not exist : ", opt.MODEL_SAVE_PATH)
        return

    # prepare test data
    print("==> loading data...")
    testset = TestSet(opt.TEST_FILE, opt=opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.TEST_BATCH_SIZE, shuffle=False)
    print("==> load data successfully")

    # do test
    net.eval()
    pred_signed_time = []
    for batch_idx, (inputs, payed_time) in enumerate(tqdm(testloader)):
        # if opt.USE_CUDA:
        #     inputs = inputs.cuda()

        (output_hour, output_day) = net(inputs.float())
        output_hour = output_hour.data.cpu().numpy()
        # transform output_day into integer
        output_day = np.floor(output_day.data.cpu().numpy() + 1 - opt.Threshold)
        # calculate pred_signed_time via output
        for i in range(len(inputs)):
            # calculate pred_signed_time via output
            pred_time_hour = np.argmax(output_hour[i])
            pred_time_day = np.argmax(output_day[i]) + 1

            temp_payed_time = payed_time[i]
            temp_payed_time = datetime.datetime.strptime(temp_payed_time, "%Y-%m-%d %H:%M:%S")
            temp_pred_signed_time = temp_payed_time + relativedelta(days = int(pred_time_day))
            temp_pred_signed_time = temp_pred_signed_time.replace(hour = int(pred_time_hour))
            pred_signed_time.append(temp_pred_signed_time.strftime('%Y-%m-%d %H'))

    # save predict result to txt file
    with open(opt.TEST_OUTPUT_PATH, 'w') as f:
        for res in pred_signed_time:
            f.write(res + "\n")

if __name__=='__main__':
    get_test_result()