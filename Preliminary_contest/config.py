import time
import os
class Config(object):
    def __init__(self):
        self.USE_CUDA = True
        self.NUM_EPOCHS = 1000
        self.USING_MODEL = True
        self.TRAIN_BATCH_SIZE = 128
        self.VAL_BATCH_SIZE = 1024
        self.TEST_BATCH_SIZE = 128
        self.TRAIN_FILE         =       './data/SeedCup_pre_train.csv'
        self.VAL_FILE           =       './data/SeedCup_pre_train.csv'
        self.TEST_FILE          =       './data/SeedCup_pre_test.csv'
        self.TEST_OUTPUT_FOLDER =       './test_output/'
        self.TEST_OUTPUT_PATH   =       self.TEST_OUTPUT_FOLDER + 'test_' + str(int(time.time())) + '.txt'

        self.MODEL_FILE_NAME    =       'model.pkl'
        self.MODEL_SAVE_FOLDER  =       './model/'
        self.MODEL_SAVE_PATH    =       self.MODEL_SAVE_FOLDER + self.MODEL_FILE_NAME
        self.LR = 1e-3  # default learning rate

        self.EMBEDDING_DIM = 200
        self.INPUT_SIZE = 11
        self.LAYER1_SIZE = 1000
        self.LAYER2_SIZE = 1000
        self.LAYER3_SIZE = 1000
        self.OUTPUT_HOUR_SIZE = 24
        self.OUTPUT_DAY_SIZE = 20

        self.uid_range = 8020516
        self.plat_form_range = 4
        self.biz_type_range = 5
        self.product_id_range = 141433
        self.cate1_id_range = 26
        self.cate2_id_range = 271
        self.cate3_id_range = 1608
        self.seller_uid_range = 999
        self.company_name_range = 950
        self.rvcr_prov_name_range = 32
        self.rvcr_city_name_range = 434

        self.val_step = 1
        self.Dataset_Normorlize = False
        self.Train_Val_ratio = 0.9

        self.mkdir()
    def get_lr(self, epoch):

        if (epoch+1) % 10 == 0:
            self.LR*=0.9
        print("epoch {}: learning rate {}".format(epoch, self.LR))
        return self.LR
    def mkdir(self):
        if not os.path.exists(self.TEST_OUTPUT_FOLDER):
            os.mkdir(self.TEST_OUTPUT_FOLDER)
        if not os.path.exists(self.MODEL_SAVE_FOLDER):
            os.mkdir(self.MODEL_SAVE_FOLDER)