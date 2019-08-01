import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.TestQueryObjectNNDataLoader import TestQueryObjectNNDataLoader, load_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils import save_checkpoint
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls, feature_transform_reguliarzer
import numpy as np

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--train_metric', type=str, default=False, help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--rotation',  default=None, help='range of training rotation')
    parser.add_argument('--model_name', default='pointnet2', help='range of training rotation')
    parser.add_argument('--feature_transform', default=False, help="use feature transform in pointnet")
    return parser.parse_args()



def getGlobalFeature( str_gorq, model, loader):
    mean_correct = []
    feature_martix = []
    path = './GlobalFeature/'
    
    for j, data in tqdm(enumerate(loader, 0), total=len(loader) ):

        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        classifier = model.eval()
        pred, trans_feat, global_feature = classifier(points)
        # print(l_ist( pred.size()))         # 32*40   32是batch_size的大小
        pred_choice = pred.data.max(1)[1]
        # print(list( pred_choice.size()))  # 32
        # print("pred_choice:",pred_choice)
        # print("target:", target)
        # print("ans:", pred_choice-target.long()) # 显示0 是预测对了的

        # print(j,list(global_feature.size()))
        # print(global_feature) 
        # feature_martix.append(global_feature)
        cpu_numpy_feature = torch.squeeze(global_feature.cpu()).detach().numpy()
        # print(k)
        filename = str_gorq + "{0}.csv".format('%d'%j)
        np.savetxt(path+filename,cpu_numpy_feature, fmt='%.10f\t', newline='\n')

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    return np.mean(mean_correct), feature_martix

def getTopK( sim_list ):
    top_k = 5
    arr = np.array(sim_list)
    top_k_idx=arr.argsort()[::-1][0:top_k]
    # print(top_k_idx)
    return list(top_k_idx)

def getSimilarity(tensor_1, tensor_2):
    # normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    # normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    # return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
    # dis = torch.sqrt(torch.sum((tensor_1-tensor_1)**2))
    # return dis
    similar = torch.cosine_similarity(tensor_1, tensor_2, dim=1) # 余弦相似度
    return similar 



def getIndexMartix( qMartix, dMartix):
    
    index_martix = []
    for qline in qMartix:
        sim_list = []
        for dline in dMartix:
            similar = getSimilarity(qline, dline)
            sim_list.append(similar)
        # 这里得到了一个Query A 与 database中所有model的一个相似度 [3308]
        index_list = getTopK(sim_list) # 返回前K大的序号
        index_martix.append(index_list)
    return index_martix


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # datapath = './data/ModelNet/'  
    datapath = './data/objecnn20_data_hdf5_2048/'
    if args.rotation is not None:
        ROTATION = (int(args.rotation[0:2]),int(args.rotation[3:5]))
    else:
        ROTATION = None

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/Test_%sObjectNNClf-'%args.model_name+ str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)


    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/test_%s_ObjectNNClf.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------Test---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    database_data, database_label, query_data, query_label = load_data(datapath, classification=True)

    print(">>>>>>>>>database_data:",database_data.shape)
    print(">>>>>>>>>query_data:",query_data.shape)

    logger.info("The number of database_data data is: %d",database_data.shape[0])
    logger.info("The number of query_data data is: %d", query_data.shape[0])
    
    ###################### 加载 database 和 query ######################
    databaseDataset = TestQueryObjectNNDataLoader(database_data, database_label, rotation=ROTATION)
    if ROTATION is not None:
        print('The range of training rotation is',ROTATION)
    queryDataset = TestQueryObjectNNDataLoader(query_data, query_label, rotation=ROTATION)

    databaseDataLoader = torch.utils.data.DataLoader(databaseDataset, batch_size=args.batchsize, shuffle=False)
    queryDataLoader  = torch.utils.data.DataLoader(queryDataset, batch_size=args.batchsize, shuffle=False) # 不打乱



    '''MODEL LOADING'''
    num_class = 20
    ###################### PointNetCls ######################
    classifier = PointNetCls(num_class,args.feature_transform).cuda() if args.model_name == 'pointnet' else PointNet2ClsMsg().cuda()
    # classifier = PointNetCls(num_class,args.feature_transform) if args.model_name == 'pointnet' else PointNet2ClsMsg()

    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Please Input the pretrained model ***.pth')
        return 

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''QueryING'''
    logger.info('Start query...')

    scheduler.step()

    ###################### Query ######################
    classifier.eval()

    _acc, database_feature_martix   = getGlobalFeature('database', classifier.eval(), databaseDataLoader )
        
    _acc, query_feature_martix      = getGlobalFeature('query', classifier.eval(), queryDataLoader    )

    # # print(query_feature_martix)
    # print("query_feature_martix: ", np.array(query_feature_martix).shape)
    # print("database_feature_martix: ", np.array(database_feature_martix).shape)


    # index_martix = getIndexMartix(query_feature_martix, database_feature_martix)
    # print(index_martix)
    # np.savetxt("index_martix.csv",index_martix,fmt='%.f\t',newline='\n')
    # print('\r Test %s: %f' % (blue('Accuracy'),acc))
    # logger.info('Test Accuracy: %f', acc)
    # logger.info('End of testing...'),

if __name__ == '__main__':
    args = parse_args()
    main(args)
