import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys, csv

def read_csv_dataset(filename):
    dataset = {}
    with open(filename, 'rt', encoding="utf-8") as fin:
        reader = csv.reader(fin)
        next(reader, None) # 跳过首行
        line = 0
        for row in reader:
            fullid = row[0]
            category = row[1]
            subcategory = row[2]
            dataset[line] = (fullid, category, subcategory)   # 'snn.098_1606786': ('Bag', 'Backpack')
            line += 1
    return dataset

def getTopK( sim_list , top_k = 5):
    arr = np.array(sim_list)
    top_k_idx=arr.argsort()[::-1][0:top_k]
    # print(top_k_idx)
    return list(top_k_idx)

def getSimilarity(vector1, vector2):
    # normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    # normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    # return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
    # dis = torch.sqrt(torch.sum((tensor_1-tensor_1)**2))
    # return dis
    # similar = torch.cosine_similarity(tensor_1, tensor_2, dim=1) # 余弦相似度
    similar = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    # print(similar)
    return similar

def getIndexMartix( qMartix, dMartix, top_k = 5 ):
    
    index_martix = []
    # for qline in qMartix:
    for _, qline in tqdm(enumerate(qMartix, 0), total=427):
        sim_list = []
        for dline in dMartix:
            similar = getSimilarity(qline, dline)
            sim_list.append(similar)
        # 这里得到了一个Query A 与 database中所有model的一个相似度 [3308]
        index_list = getTopK(sim_list, top_k) # 返回前K大的序号
        index_martix.append(index_list)
    return index_martix

def getIndexMartixByClass( qMartix, dMartix, query_csv, shapenet_csv, top_k = 5 ):
    index_martix = []
    # for qline in qMartix:
    for _, (qline,csv_line) in tqdm(enumerate( zip(qMartix,query_csv), 0), total=427):
        # print(qline,)
        sim_list = []
        for dline, s_line in zip(dMartix, shapenet_csv):
            if( query_csv[csv_line][1] == shapenet_csv[s_line][1]):
                # print(query_csv[csv_line][1], shapenet_csv[s_line][1])
                similar = getSimilarity(qline, dline)
                sim_list.append(similar)
            else:
                sim_list.append(0)
        # 这里得到了一个Query A 与 database中所有model的一个相似度 [3308]
        index_list = getTopK(sim_list, top_k) # 返回前K大的序号
        index_martix.append(index_list)
    return index_martix

if __name__ == '__main__':

    print("----------Begin Load database.csv and query.csv ----------")

    database = np.loadtxt('./GlobalFeature/database0.csv')
    for i in range(1,104):
        new_data = np.loadtxt('./GlobalFeature/database{0}.csv'.format('%d'%i))
        database = np.concatenate([database ,new_data])

    query = np.loadtxt('./GlobalFeature/query0.csv')
    for i in range(1,14):
        new_data = np.loadtxt('./GlobalFeature/query{0}.csv'.format('%d'%i))
        query = np.concatenate([query ,new_data])
    print("----------Load successfully----------")
    print("Begin calculate....")

    start = time.time()
    top_k = 10
    index_martix = getIndexMartix(query, database, top_k)
    
    # path = '/home/sun/WorkSpace/PointDeal/Pointnet_Pointnet2_pytorch/Result/'
    # query_csv = read_csv_dataset(path+'test_answer.csv')    # 423: ('snn.246_431617b', 'Table', 'SideTable')
    # shapenet_csv = read_csv_dataset(path+'shapenet.csv')    # 3305: ('wss.ffb4e07d613b6a62b...', 'Box', 'TissueBox')
    # # print(shapenet_csv)
    # index_martix = getIndexMartixByClass(query, database, query_csv, shapenet_csv, top_k)

    end = time.time()
    print("Time :%.2f s"%(end-start))

    arr_index_martix = np.array(index_martix)
    print("arr_index_martix size:",arr_index_martix.shape)


    path = './Result/'
    filename = 'indexMartix-Top%d.csv'%top_k
    ### 注意这里的 delimiter 默认是 = ' ' 这样生成的数据 2853, 3304, 966, 3006, 1881,
    ### 用csv_reader读取出来之后  ['787', ' 1790', ' 703', ' 1861']
    np.savetxt(path+filename,arr_index_martix, fmt='%.f,', delimiter='') 
    print("----------Save %s----------"%(path+filename))
