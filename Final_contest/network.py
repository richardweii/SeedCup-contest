import sys
import os
sys.path.append(os.getcwd())
import torch.nn.functional
import torch.nn as nn
import torch
'''
design loss function
'''
class My_unbalance_loss1(nn.Module):
    def __init__(self, weight):
        super(My_unbalance_loss1,self).__init__()
        self.weight = weight
    def forward(self, output, label):
        pre_result = torch.argmax(output, dim=1)
        weight = torch.gt(pre_result, label).view([-1,1]).float()
        weight = torch.add(weight, 1/49)
        one_hot = torch.zeros(label.shape[0], 20).cuda()
        one_hot = one_hot.scatter(1,label.view([-1,1]), 1)
        output_probility = torch.nn.functional.log_softmax(output, dim=1)
        cross_entropy = -one_hot * output_probility * weight * self.weight
        return torch.sum(cross_entropy)

class My_mse_loss(nn.Module):
    def __init__(self):
        super(My_mse_loss,self).__init__()
    def forward(self, output, label):
        label = label.view([-1, 1])
        loss = torch.nn.functional.leaky_relu(output - label.float(), negative_slope=-0.5)
        return torch.sum(loss**2)

class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.encoder_payed_time = nn.Embedding(24, opt.EMBEDDING_DIM)
        self.encoder_plat_form = nn.Embedding(opt.plat_form_range, opt.EMBEDDING_DIM)
        self.encoder_biz_type = nn.Embedding(opt.biz_type_range, opt.EMBEDDING_DIM)
        self.encoder_product_id = nn.Embedding(opt.product_id_range, opt.EMBEDDING_DIM)

        self.encoder_cate1_id = nn.Embedding(opt.cate1_id_range, opt.EMBEDDING_DIM)
        self.encoder_cate2_id = nn.Embedding(opt.cate2_id_range, opt.EMBEDDING_DIM)
        self.encoder_cate3_id = nn.Embedding(opt.cate3_id_range, opt.EMBEDDING_DIM)

        self.encoder_seller_uid = nn.Embedding(opt.seller_uid_range, opt.EMBEDDING_DIM)
        self.encoder_company_name = nn.Embedding(opt.company_name_range, opt.EMBEDDING_DIM)
        self.encoder_rvcr_prov_name = nn.Embedding(opt.rvcr_prov_name_range, opt.EMBEDDING_DIM)
        self.encoder_rvcr_city_name = nn.Embedding(opt.rvcr_city_name_range, opt.EMBEDDING_DIM)

        self.full_conection_layer_hour = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(opt.INPUT_SIZE* opt.EMBEDDING_DIM, opt.LAYER1_SIZE),
            nn.BatchNorm1d(opt.LAYER1_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(opt.LAYER1_SIZE, opt.LAYER2_SIZE),
            nn.BatchNorm1d(opt.LAYER2_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(opt.LAYER2_SIZE, opt.LAYER3_SIZE),
            nn.BatchNorm1d(opt.LAYER3_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(opt.LAYER3_SIZE, opt.OUTPUT_HOUR_SIZE),
        )
        self.full_conection_layer_day = nn.Sequential(
            #  TODO change input dimension
            nn.Linear(opt.INPUT_SIZE* opt.EMBEDDING_DIM, opt.LAYER1_SIZE),
            nn.BatchNorm1d(opt.LAYER1_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(opt.LAYER1_SIZE, opt.LAYER2_SIZE),
            nn.BatchNorm1d(opt.LAYER2_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(opt.LAYER2_SIZE, opt.LAYER3_SIZE),
            nn.BatchNorm1d(opt.LAYER3_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(opt.LAYER3_SIZE, opt.OUTPUT_DAY_SIZE),
        )

    def forward(self, x):
        '''
        embedding layers
        '''
        output_encoder_payed_time = self.encoder_payed_time(x[:, 0].long())
        output_encoder_plat_form = self.encoder_plat_form(x[:, 1].long())
        output_encoder_biz_type = self.encoder_biz_type(x[:, 2].long())
        output_encoder_product_id = self.encoder_product_id(x[:, 3].long())

        output_encoder_cate1_id = self.encoder_cate1_id(x[:, 4].long())
        output_encoder_cate2_id = self.encoder_cate2_id(x[:, 5].long())
        output_encoder_cate3_id = self.encoder_cate3_id(x[:, 6].long())
        output_encoder_seller_uid = self.encoder_seller_uid(x[:, 7].long())

        output_encoder_company_name = self.encoder_company_name(x[:, 8].long())
        output_encoder_rvcr_prov_name = self.encoder_rvcr_prov_name(x[:, 9].long())
        output_encoder_rvcr_city_name = self.encoder_rvcr_city_name(x[:, 10].long())

        concat_encoder_output = torch.cat((output_encoder_payed_time,output_encoder_plat_form,
                                           output_encoder_biz_type, output_encoder_product_id,
                                           output_encoder_cate1_id, output_encoder_cate2_id,
                                           output_encoder_cate3_id, output_encoder_seller_uid,
                                           output_encoder_company_name, output_encoder_rvcr_prov_name,
                                           output_encoder_rvcr_city_name
                                           ), 1)

        '''
        Fully Connected layers
        you can attempt muti-task through uncommenting the following code and modifying related code in train()
        '''
        output_FC_hour = self.full_conection_layer_hour(concat_encoder_output)
        output_FC_day = self.full_conection_layer_day(concat_encoder_output)
        return output_FC_hour, output_FC_day