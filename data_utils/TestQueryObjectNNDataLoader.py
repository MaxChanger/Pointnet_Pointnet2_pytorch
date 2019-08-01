import numpy as np
import warnings
import h5py
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def load_data(dir,classification = False):
    data_shapenet, label_shapenet, Seglabel_shapenet  = load_h5(dir + 'ply_data_train_wss3308.h5') #ply_data_train_wss3308
    data_query, label_query, Seglabel_query = load_h5(dir + 'ply_data_test_snn427_2.h5')

    train_data = np.concatenate([data_shapenet])
    train_label = np.concatenate([label_shapenet])
    train_Seglabel = np.concatenate([Seglabel_shapenet])

    test_data = np.concatenate([data_query])
    test_label = np.concatenate([label_query])
    test_Seglabel = np.concatenate([Seglabel_query])

    train_data = train_data.astype(np.float32)
    train_label = train_label.astype(np.uint8)
    test_data = test_data.astype(np.float32)
    test_label = test_label.astype(np.uint8)

    if classification:
        return train_data, train_label, test_data, test_label
        # return train_data[:100], train_label[:100], test_data[:5], test_label[:5]
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel

class TestQueryObjectNNDataLoader(Dataset):
    def __init__(self, data, labels, rotation = None):
        self.data = data
        self.labels = labels
        self.rotation = rotation

    def __len__(self):
        return len(self.data)

    def rotate_point_cloud_by_angle(self, data, rotation_angle):
        """
        Rotate the point cloud along up direction with certain angle.
        :param batch_data: Nx3 array, original batch of point clouds
        :param rotation_angle: range of rotation
        :return:  Nx3 array, rotated batch of point clouds
        """
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data, rotation_matrix)

        return rotated_data

    def __getitem__(self, index):
        if self.rotation is not None:
            pointcloud = self.data[index]
            angle = np.random.randint(self.rotation[0], self.rotation[1]) * np.pi / 180
            pointcloud = self.rotate_point_cloud_by_angle(pointcloud, angle)

            return pointcloud, self.labels[index]
        else:
            return self.data[index], self.labels[index]