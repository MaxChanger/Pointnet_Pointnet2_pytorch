import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data
# from data_utils.S3DISDataLoader import S3DISDataLoader, recognize_all_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils import save_checkpoint
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls, feature_transform_reguliarzer

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



def test(model, loader):
    mean_correct = []
    for j, data in enumerate(loader, 0):
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

        # print(list(global_feature.size()))
        print(global_feature) 


        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    return np.mean(mean_correct)


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # datapath = './data/ModelNet/'  
    datapath = './data/modelnet40_ply_hdf5_2048/'
    if args.rotation is not None:
        ROTATION = (int(args.rotation[0:2]),int(args.rotation[3:5]))
    else:
        ROTATION = None

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path('./experiment/checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = Path('./experiment/logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("PointNet2")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('./experiment/logs/test_%s_'%args.model_name+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------Test---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])
    trainDataset = ModelNetDataLoader(train_data, train_label, rotation=ROTATION)
    if ROTATION is not None:
        print('The range of training rotation is',ROTATION)
    testDataset = ModelNetDataLoader(test_data, test_label, rotation=ROTATION)

    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchsize, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False) # 不打乱

    '''MODEL LOADING'''
    num_class = 40
    ###################### PointNetCls ######################
    classifier = PointNetCls(num_class,args.feature_transform).cuda() if args.model_name == 'pointnet' else PointNet2ClsMsg().cuda()
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

    '''TestING'''
    logger.info('Start testing...')

    scheduler.step()

    acc = test(classifier.eval(), testDataLoader)

    print('\r Test %s: %f' % (blue('Accuracy'),acc))
    logger.info('Test Accuracy: %f', acc)

    logger.info('End of testing...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
