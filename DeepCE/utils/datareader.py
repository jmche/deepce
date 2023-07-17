from .data_utils import *

seed = 343
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)


class DataReader(object):
    def __init__(self, drug_file, gene_file, cell_file, data_file_train, data_file_dev, data_file_test, data_file_predict,
                 filter, device):
        self.device = device
        self.drug, self.drug_dim = read_drug_string(drug_file)
        self.gene = read_gene(gene_file, self.device)
        cell = read_cell(cell_file, self.device)
        feature_train, label_train, colName = read_data(data_file_train, filter, dataType = 'Train')
        feature_dev, label_dev, colName = read_data(data_file_dev, filter, dataType = 'Dev')
        feature_test, label_test, colName = read_data(data_file_test, filter, dataType = 'Test')
        feature_predict, label_predict, colName = read_data(data_file_predict, filter, dataType = 'Predict')
        self.feature_train = feature_train
        self.feature_dev = feature_dev
        self.feature_test = feature_test
        self.feature_predict = feature_predict
        self.colName = colName
        self.train_feature, self.dev_feature, self.test_feature, self.predict_feature, self.train_label, \
        self.dev_label, self.test_label, self.predict_label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            transfrom_to_tensor(cell, feature_train, label_train, feature_dev, label_dev,
                                           feature_test, label_test, feature_predict, label_predict, self.drug, self.device)

    def get_batch_data(self, dataset, batch_size, shuffle):
        if dataset == 'train':
            feature = self.train_feature
            label = self.train_label
            featureInfo = self.feature_train
        elif dataset == 'dev':
            feature = self.dev_feature
            label = self.dev_label
            featureInfo = self.feature_dev
        elif dataset == 'test':
            feature = self.test_feature
            label = self.test_label
            featureInfo = self.feature_test
        elif dataset == 'predict':
            feature = self.predict_feature
            label = self.predict_label
            featureInfo = self.feature_predict
        if shuffle:
            index = torch.randperm(len(feature['drug'])).long()
            index = index.numpy()
        for start_idx in range(0, len(feature['drug']), batch_size):
            if shuffle:
                excerpt = index[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            output = dict()
            # output['drug'] = convert_smile_to_feature(feature['drug'][excerpt], self.device)
            # output['mask'] = create_mask_feature(output['drug'], self.device)
            output['drug'] = feature['drug'][excerpt]
            if self.use_pert_type:
                output['pert_type'] = feature['pert_type'][excerpt]
            if self.use_cell_id:
                output['cell_id'] = feature['cell_id'][excerpt]
            if self.use_pert_idose:
                output['pert_idose'] = feature['pert_idose'][excerpt]
            yield output, label[excerpt], featureInfo[excerpt], self.colName
