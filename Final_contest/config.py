import time
import os
'''
uid                            8020516
plat_form                            4
biz_type                             5
create_time        2019-04-30 23:59:59
payed_time         2019-05-01 22:40:54
product_id                      141433
cate1_id                            26
cate2_id                           271
cate3_id                          1608
seller_uid                         999
company_name                       950
lgst_company                        16
warehouse_id                        11
shipped_prov_id                     27
shipped_city_id                    117
rvcr_prov_name                      32
rvcr_city_name                     434
shipped_time       2019-06-14 19:31:02
got_time           2019-06-14 19:25:56
dlved_time         2019-06-14 20:30:30
signed_time        2019-06-17 10:23:23
'''
class Config(object):
    def __init__(self):
        self.USE_CUDA = True
        self.NUM_EPOCHS = 100
        self.USING_MODEL = True
        self.TRAIN_BATCH_SIZE = 256
        self.VAL_BATCH_SIZE = 1024
        self.TEST_BATCH_SIZE = 256
        self.TRAIN_FILE         =       './data/SeedCup_final_train.csv'
        self.VAL_FILE           =       './data/SeedCup_final_train.csv'
        self.TEST_FILE          =       './data/SeedCup_final_test.csv'
        self.TEST_OUTPUT_FOLDER =       './test_output/'
        self.TEST_OUTPUT_PATH   =       self.TEST_OUTPUT_FOLDER + 'test_' + str(int(time.time())) + '.txt'

        self.MODEL_FILE_NAME    =       'model.pkl'
        self.MODEL_SAVE_FOLDER  =       './model/'
        self.MODEL_SAVE_PATH    =       self.MODEL_SAVE_FOLDER + self.MODEL_FILE_NAME
        self.LR = 1e-4 # default learning rate

        self.EMBEDDING_DIM = 100
        self.INPUT_SIZE = 11
        self.LAYER1_SIZE = 500
        self.LAYER2_SIZE = 500
        self.LAYER3_SIZE = 500

        self.OUTPUT_HOUR_SIZE = 24
        self.OUTPUT_DAY_SIZE = 1

        self.uid_range = 8020517
        self.plat_form_range = 5
        self.biz_type_range = 6
        self.product_id_range = 141434
        self.cate1_id_range = 27
        self.cate2_id_range = 272
        self.cate3_id_range = 1609
        self.seller_uid_range = 1000
        self.company_name_range = 951
        self.rvcr_prov_name_range = 33
        self.rvcr_city_name_range = 435

        self.val_step = 1
        self.Dataset_Normorlize = False
        self.Train_Val_ratio = 9
        self.Threshold = 0.5
        self.mkdir()
    def get_lr(self, epoch):

        if epoch != 0:
            self.LR*=0.95
        print("epoch {}: learning rate {}".format(epoch, self.LR))
        return self.LR
    def mkdir(self):
        if not os.path.exists(self.TEST_OUTPUT_FOLDER):
            os.mkdir(self.TEST_OUTPUT_FOLDER)
        if not os.path.exists(self.MODEL_SAVE_FOLDER):
            os.mkdir(self.MODEL_SAVE_FOLDER)