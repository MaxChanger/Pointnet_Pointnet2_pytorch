import os,shutil
import sys, csv
def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print(srcfile+"not exist!")
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print( "copy" + srcfile + "->" + dstfile )

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

def read_indexMartix(filename, top_k = 5):
    dataset = {}
    with open(filename, 'rt', encoding="utf-8") as fin:
        reader = csv.reader(fin)
        # next(reader, None) # 跳过首行
        line = 0
        for row in reader:
            dataset[line] = ( row[:top_k] )   # 'snn.098_1606786': ('Bag', 'Backpack')
            line += 1
    return dataset




if __name__ == '__main__':
    
    path = '/home/sun/WorkSpace/PointDeal/Pointnet_Pointnet2_pytorch/Result/'

    query_csv = read_csv_dataset(path+'test_answer.csv')    # 423: ('snn.246_431617b', 'Table', 'SideTable')
    shapenet_csv = read_csv_dataset(path+'shapenet.csv')    # 3305: ('wss.ffb4e07d613b6a62bbfa57fc7493b378', 'Box', 'TissueBox')
    indexMartix = read_indexMartix(path+'indexMartix-Top10.csv',top_k = 10) # # 423: ['1423', ' 2762', ' 2649', ' 2029', ' 3044', ' 2129']

    for key in range(200,215): # indexMartix
        file_dir = path + "query_{0}_{1}_{2}/".format('%03d'%key,'%s'%query_csv[key][1],'%s'%query_csv[key][2])
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir+"/model/")
        
        objname =  query_csv[key][0][4:]+ ".ply"
        rootpath = '/home/sun/WorkSpace/shrec17_data_jan27/scenenn/objects/'
        srcfile = rootpath + objname
        dstfile = file_dir + objname
        mycopyfile(srcfile,dstfile)

        # print(indexMartix[key]) # ['787', ' 1790', ' 703', ' 1861', ' 1970', ' 1964', ' 982', ' 2348', ' 1560', ' 286']
        top = 1
        for item in indexMartix[key]:
            # print(item)
            objname =  shapenet_csv[int(item)][0][4:] + ".obj"
            rootpath = '/home/sun/WorkSpace/shrec17_data_jan27/shapenet/models/'
            srcfile = rootpath + objname
            dstfile = file_dir + "model/{0}_{1}_{2}".format('%02d'%top,'%s'%shapenet_csv[int(item)][1], '%s'%shapenet_csv[int(item)][2] ) + ".obj"
            top += 1
            # print(dstfile)
            mycopyfile(srcfile,dstfile)