# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install lightning==2.0.3 -c conda-forge
# pip3 install pandas rdkit torch-geometric wandb
import os, subprocess
import sys, re
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import argparse
from models import DeepCE
from utils import DataReader
from utils import rmse, correlation, precision_k, cosine
import pandas as pd
if sys.platform.startswith('win'):
    splitSig = '\\'
else:
    splitSig = '/'
start_time = datetime.now()
current_file_path = os.path.abspath(__file__)
rootPath = current_file_path.rsplit(splitSig,1)[0]
parser = argparse.ArgumentParser(description='DeepCE Training')
parser.add_argument('--gpuID', nargs='+', default=[0], help='List of GPU')
parser.add_argument('--specificCell', nargs='+', default=[], help='remove cells from train and choose the cells from dev and test data set')
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--batch_size', default=16)
parser.add_argument('--max_epoch', default=800)
parser.add_argument('--attentionType', nargs='+', default=[], help='List of attention types')
parser.add_argument('--drug_file', default=os.path.join(rootPath, 'data', 'drugs_smiles.csv'))
parser.add_argument('--gene_file', default=os.path.join(rootPath, 'data', 'gene_vector.csv'))
parser.add_argument('--cell_file', default=os.path.join(rootPath, 'data', 'cell_id_features.csv'))
parser.add_argument('--predictFile', default=os.path.join(rootPath, 'data', 'smilesPrediction.csv'))
parser.add_argument('--train_file', default=os.path.join(rootPath, 'data', 'uniqueDrug24H_RemoveDupCondition', 'signature_train.csv'))
parser.add_argument('--dev_file', default=os.path.join(rootPath, 'data', 'uniqueDrug24H_RemoveDupCondition', 'signature_dev.csv'))
parser.add_argument('--test_file', default=os.path.join(rootPath, 'data', 'uniqueDrug24H_RemoveDupCondition', 'signature_test.csv'))
parser.add_argument('--dataSource', default='')

args = parser.parse_args()
gpuID = args.gpuID
specificCell = args.specificCell
dropout = float(args.dropout)
batch_size = int(args.batch_size)
max_epoch = int(args.max_epoch)

attentionType = args.attentionType
drug_file = args.drug_file
gene_file = args.gene_file
cell_file = args.cell_file
gene_expression_file_prediction = args.predictFile
gene_expression_file_train = args.train_file
gene_expression_file_dev = args.dev_file
gene_expression_file_test = args.test_file
dataSource = args.dataSource
if(len(dataSource) == 0):
    dataSource = gene_expression_file_train.rsplit(splitSig)[-2]
else:
    gene_expression_file_train = splitSig.join(gene_expression_file_train.rsplit(splitSig)[:-2] + [dataSource] + [gene_expression_file_train.rsplit(splitSig)[-1]])
    gene_expression_file_dev = splitSig.join(gene_expression_file_dev.rsplit(splitSig)[:-2] + [dataSource] + [gene_expression_file_dev.rsplit(splitSig)[-1]])
    gene_expression_file_test = splitSig.join(gene_expression_file_test.rsplit(splitSig)[:-2] + [dataSource] + [gene_expression_file_test.rsplit(splitSig)[-1]])

def get_gpu_temperatures():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True)
        temperatures = result.stdout.strip().split('\n')
        return [str(temp) for temp in temperatures]
    except FileNotFoundError:
        return []

def get_gpu_count():
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        gpu_info = result.stdout.strip().split('\n')
        return len(gpu_info)
    except FileNotFoundError:
        return 0
num_gpus = get_gpu_count()
print("Number of available GPUs:", num_gpus)

if num_gpus > 0:
    if(len(gpuID) == 0):
        gpuID = [i for i in range(torch.cuda.device_count())]
        print('Run with all GPUs since gpuID list is not specified.')
    else:
        print('Run with GPU: ' + ",".join([str(i) for i in gpuID]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpuID])
    device = torch.device("cuda")
else:
    print('Run with CPU.')
    device = torch.device("cpu")

print('Device:', device)
trainCell = []
with open(cell_file, 'r') as f:
    colName = f.readline()  # skip header
    colName = colName.strip().split(',')
    for line in f:
        line = line.strip().split(',')
        trainCell.append(line[0])
trainCell = list(set(trainCell))

# parameters initialization
if(len(attentionType) == 0):
    attentionType = ['Drug', 'Drug_Gene']
    attentionType = ['Drug', 'Cell', 'Gene', 'Drug_Gene', 'Cell_Gene', 'Drug_Cell']
drug_input_dim = {'atom': 62, 'bond': 6}
drug_embed_dim = 128
gene_embed_dim = 128
cell_embed_dim = 128
pert_type_emb_dim = 4
pert_idose_emb_dim = 4
hid_dim = 128
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
num_gene = 978
n_layers = 2
n_heads = 4
pf_dim = 512
precision_degree = [10, 20, 50, 100]
loss_type = 'point_wise_mse' # 'point_wise_mse' # 'pearsonNcosine'  #'point_wise_mse'
initializer = torch.nn.init.xavier_uniform_
cellListAll = ['A375', 'A549', 'ASC', 'ASC.C', 'BT20', 'CD34', 'HA1E', 'HCC515', 'HELA', 'HEPG2', 'HME1', 'HS578T', 'HT29', 'HUES3', 'HUVEC', 'JURKAT', 'LNCAP', 'MCF10A', 'MCF7', 'MDAMB231', 'MNEU.E', 'NEU', 'NPC', 'NPC.CAS9', 'NPC.TAK', 'PC3', 'SKBR3', 'SKL', 'SKL.C', 'YAPC']
cellListNonNormal = ['A375', 'A549', 'ASC', 'ASC.C', 'BT20', 'HELA', 'HEPG2', 'HS578T', 'HT29', 'JURKAT', 'LNCAP', 'MCF7', 'MDAMB231', 'NPC', 'NPC.CAS9', 'NPC.TAK', 'PC3', 'SKBR3', 'SKL', 'SKL.C', 'YAPC']
cellList = ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC']
filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
          "cell_id": [cellI for cellI in cellListNonNormal if cellI in trainCell],
          "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"],
          "specificCell": []}
if(len(specificCell)):
    filter['specificCell'] = specificCell

data = DataReader(drug_file, gene_file, cell_file, gene_expression_file_train, gene_expression_file_dev,
                  gene_expression_file_test, gene_expression_file_prediction, filter, device)

numTrain = '#Train: %d' % len(data.train_feature['drug'])
numDev = '#Dev: %d' % len(data.dev_feature['drug'])
numTest = '#Test: %d' % len(data.test_feature['drug'])
numPredict = '#Predict: %d' % len(data.predict_feature['drug'])
print(numTrain)
print(numDev)
print(numTest)
print(numPredict)


# model creation
model = DeepCE(drug_input_dim=drug_input_dim, drug_emb_dim=drug_embed_dim,
                      conv_size=conv_size, degree=degree, attentionType = attentionType, 
                      gene_input_dim=np.shape(data.gene)[1], gene_emb_dim=gene_embed_dim, num_gene=np.shape(data.gene)[0], 
                      cell_input_dim = np.shape(data.train_feature['cell_id'])[1], cell_emb_dim=cell_embed_dim, 
                      n_layers = n_layers, n_heads = n_heads, pf_dim = pf_dim, 
                      hid_dim=hid_dim, dropout=dropout,
                      loss_type=loss_type, device=device, initializer=initializer,
                      pert_type_input_dim=len(filter['pert_type']), 
                      pert_idose_input_dim=len(filter['pert_idose']), pert_type_emb_dim=pert_type_emb_dim,
                      pert_idose_emb_dim=pert_idose_emb_dim,
                      use_pert_type=data.use_pert_type, use_cell_id=data.use_cell_id,
                      use_pert_idose=data.use_pert_idose)

model = model.double()
if(torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)
model.to(device)
# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
best_dev_loss = float("inf")
best_dev_pearson = float("-inf")
pearson_list_dev = []
pearson_list_test = []
spearman_list_dev = []
spearman_list_test = []
rmse_list_dev = []
rmse_list_test = []
precisionk_list_dev = []
precisionk_list_test = []
pearson_raw_list = []

drugL = {'Train':[], 'Dev': [], 'Test': []}

performanceDevList = []
performanceTestList = []
performanceList = []
if(len(specificCell)):
    saveModelName = '__'.join(attentionType) + splitSig + 'trainedM_' + loss_type + '_batch' + str(batch_size) + '_' + dataSource + '_SpecificCell_' + '_'.join(specificCell) + re.sub('\D', '_', re.sub('\\..*', '', str(datetime.now())))
else:
    saveModelName = '__'.join(attentionType) + splitSig + 'trainedM_' + loss_type + '_batch' + str(batch_size) + '_' + dataSource + '_' + re.sub('\D', '_', re.sub('\\..*', '', str(datetime.now())))

for epoch in range(max_epoch):
    print("Iteration %d:" % (epoch+1))
    print(saveModelName)
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=True)):
        ft, lb, featureInfo, colName= batch
        drug = ft['drug']
        drugIndex = torch.IntTensor([drugI for drugI in range(len(ft['drug']))])
        if data.use_pert_type:
            pert_type = ft['pert_type']
        else:
            pert_type = None
        if data.use_cell_id:
            cell_id = ft['cell_id']
        else:
            cell_id = None
        if data.use_pert_idose:
            pert_idose = ft['pert_idose']
        else:
            pert_idose = None
        optimizer.zero_grad()
        predict = model(drugIndex, drug, data.gene, pert_type, cell_id, pert_idose)
        if(torch.cuda.device_count() > 1):
            loss = model.module.loss(lb, predict)
        else:
            loss = model.loss(lb, predict)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('Train loss:')
    print(epoch_loss/(i+1))

    model.eval()

    epoch_loss = 0
    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])

    performanceDev = [0, 0, 0, 0, 0]
    with torch.no_grad():
        for i, batch in enumerate(data.get_batch_data(dataset='dev', batch_size=batch_size, shuffle=False)):
            ft, lb, featureInfo, colName= batch
            drug = ft['drug']
            drugIndex = torch.IntTensor([drugI for drugI in range(len(ft['drug']))])
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            predict = model(drugIndex, drug, data.gene, pert_type, cell_id, pert_idose)
            if(torch.cuda.device_count() > 1):
                loss = model.module.loss(lb, predict)
            else:
                loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
        print('Dev loss:')
        print(epoch_loss / (i + 1))
        rmse_score = rmse(lb_np, predict_np)
        rmse_list_dev.append(rmse_score)
        print('RMSE: %.4f' % rmse_score)
        cosine_score, _ = cosine(lb_np, predict_np)
        pearson, _ = correlation(lb_np, predict_np, 'pearson')
        pearson_list_dev.append(pearson)
        print('Pearson\'s correlation: %.4f' % pearson)
        spearman, _ = correlation(lb_np, predict_np, 'spearman')
        spearman_list_dev.append(spearman)
        print('Spearman\'s correlation: %.4f' % spearman)
        precision = []
        for k in precision_degree:
            precision_neg, precision_pos = precision_k(lb_np, predict_np, k)
            print("Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Precision@%d Negative: %.4f" % (k, precision_neg))
            precision.append([precision_pos, precision_neg])
        precisionk_list_dev.append(precision)
        if best_dev_pearson < pearson:
            best_dev_pearson = pearson

        performanceDev = [pearson, spearman, cosine_score, rmse_score] + list(np.array(precision).flatten())

    performanceTest = [0, 0, 0, 0, 0]
    epoch_loss = 0
    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    with torch.no_grad():
        for i, batch in enumerate(data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False)):
            ft, lb, featureInfo, colName= batch
            drug = ft['drug']
            drugIndex = torch.IntTensor([drugI for drugI in range(len(ft['drug']))])
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            predict = model(drugIndex, drug, data.gene, pert_type, cell_id, pert_idose)
            if(torch.cuda.device_count() > 1):
                loss = model.module.loss(lb, predict)
            else:
                loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)

        print('Test loss:')
        print(epoch_loss / (i + 1))
        rmse_score = rmse(lb_np, predict_np)
        rmse_list_test.append(rmse_score)
        print('RMSE: %.4f' % rmse_score)
        cosine_score, _ = cosine(lb_np, predict_np)
        pearson, _ = correlation(lb_np, predict_np, 'pearson')
        pearson_list_test.append(pearson)
        print('Pearson\'s correlation: %.4f' % pearson)
        spearman, _ = correlation(lb_np, predict_np, 'spearman')
        spearman_list_test.append(spearman)
        print('Spearman\'s correlation: %.4f' % spearman)
        precision = []
        for k in precision_degree:
            precision_neg, precision_pos = precision_k(lb_np, predict_np, k)
            print("Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Precision@%d Negative: %.4f" % (k, precision_neg))
            precision.append([precision_pos, precision_neg])
        precisionk_list_test.append(precision)

        performanceTest = [pearson, spearman, cosine_score, rmse_score] + list(np.array(precision).flatten())

    if ((len(performanceDevList) < 5 or performanceDev[0] > performanceDevList[0][0]) or (len(performanceTestList) < 5 or performanceTest[0] > performanceTestList[0][0]) or (len(performanceList) < 5 or (performanceDev[0] + performanceTest[0]) > performanceList[0][0])) and epoch > 50:
        # 保存当前模型
        performanceDevList.append(performanceDev)
        performanceDevList.sort(reverse=True)
        performanceDevList = performanceDevList[:5]

        performanceTestList.append(performanceTest)
        performanceTestList.sort(reverse=True)
        performanceTestList = performanceTestList[:5]

        performanceList.append(list(np.array(performanceDev) + np.array(performanceTest)))
        performanceList.sort(reverse=True)
        performanceList = performanceList[:5]

        if((epoch + 1) == 51):
            torch.save(model, os.path.join(rootPath, 'trainedModel', saveModelName, f'model_{epoch}.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(rootPath, 'trainedModel', saveModelName, f'model_{epoch}.pth'))

        performanceData = pd.DataFrame([['Dev', epoch] + performanceDev, ['Test', epoch] + performanceTest])
        PrecisionName = [['Precision' + str(degreeK) + 'Pos', 'Precision' + str(degreeK) + 'Neg'] for degreeK in precision_degree]
        performanceData.columns = ['Data', 'Epoch', 'pearson', 'spearman', 'cosine_similarity', 'rmse_score'] + list(np.array(PrecisionName).flatten())
        if(not os.path.exists(os.path.join(rootPath, 'trainedModel', saveModelName, 'performance.csv'))):
            performanceData.to_csv(os.path.join(rootPath, 'trainedModel', saveModelName, 'performance.csv'), header=True, index=False)
        else:
            performanceData.to_csv(os.path.join(rootPath, 'trainedModel', saveModelName, 'performance.csv'), header=False, index=False, mode='a')

        epoch_loss = 0
        lb_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, batch in enumerate(data.get_batch_data(dataset='predict', batch_size=batch_size, shuffle=False)):
                ft, lb, featureInfo, colName= batch
                drug = ft['drug']
                drugIndex = torch.IntTensor([drugI for drugI in range(len(ft['drug']))])
                if data.use_pert_type:
                    pert_type = ft['pert_type']
                else:
                    pert_type = None
                if data.use_cell_id:
                    cell_id = ft['cell_id']
                else:
                    cell_id = None
                if data.use_pert_idose:
                    pert_idose = ft['pert_idose']
                else:
                    pert_idose = None
                predict = model(drugIndex, drug, data.gene, pert_type, cell_id, pert_idose)
                if(torch.cuda.device_count() > 1):
                    loss = model.module.loss(lb, predict)
                else:
                    loss = model.loss(lb, predict)
                epoch_loss += loss.item()
                lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
                predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)

                writingData = pd.DataFrame(predict.cpu().numpy())
                writingData = pd.concat([pd.DataFrame(featureInfo), writingData], axis=1)
                writingData.columns = colName[1:]
                if(i == 0):
                    writingData.to_csv(os.path.join(rootPath, 'trainedModel', saveModelName, 'prediction_' + str(epoch) + '.csv'), header=True, index=False)
                else:
                    writingData.to_csv(os.path.join(rootPath, 'trainedModel', saveModelName, 'prediction_' + str(epoch) + '.csv'), header=False, index=False, mode='a')
    if(not os.path.exists(os.path.join(rootPath, 'trainedModel', saveModelName))):
        os.makedirs(os.path.join(rootPath, 'trainedModel', saveModelName))
    if epoch == 0:
        with open(os.path.join(rootPath, 'trainedModel', saveModelName, 'parameter.txt'), 'w') as writeFile:
            usedParameter = ['numTrain', 'numDev', 'numTest', 'numPredict', 
                            'max_epoch', 'batch_size', 'dropout', 'device', 'loss_type', 'attentionType', 
                            'drug_input_dim', 'drug_embed_dim', 'gene_embed_dim', 'cell_embed_dim', 
                            'pert_type_emb_dim', 'pert_idose_emb_dim', 
                            'hid_dim', 'conv_size', 'degree', 'num_gene', 'n_layers', 'n_heads', 'pf_dim', 
                            'precision_degree', 'filter', 
                            'dataSource', 'gene_expression_file_train', 'gene_expression_file_dev',
                            'gene_expression_file_test', 'gene_expression_file_prediction']
            for parameterI in usedParameter:
                writeFile.write(parameterI + ': ' + str(eval(parameterI)) + '\n')
            writeFile.write('\n\nTotal params: ' + str(sum(p.numel() for p in model.parameters())) + '\n')
            writeFile.write('Trainable params: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + '\n\n')
            writeFile.write('Model: \n' + str(eval('model')) + '\n')
    performanceData = pd.DataFrame([[re.sub('\\..*', '', str(datetime.now())), re.sub('\\..*', '', str(datetime.now() - start_time)), ','.join(get_gpu_temperatures()), 'Dev', epoch] + performanceDev, [re.sub('\\..*', '', str(datetime.now())), re.sub('\\..*', '', str(datetime.now() - start_time)), ','.join(get_gpu_temperatures()), 'Test', epoch] + performanceTest])
    PrecisionName = [['Precision' + str(degreeK) + 'Pos', 'Precision' + str(degreeK) + 'Neg'] for degreeK in precision_degree]
    performanceData.columns = ['Date', 'ExecutionTime', 'GPU_Temp', 'Data', 'Epoch', 'pearson', 'spearman', 'cosine_similarity', 'rmse_score'] + list(np.array(PrecisionName).flatten())
    if(not os.path.exists(os.path.join(rootPath, 'trainedModel', saveModelName, 'performanceAll.csv'))):
        performanceData.to_csv(os.path.join(rootPath, 'trainedModel', saveModelName, 'performanceAll.csv'), header=True, index=False)
    else:
        performanceData.to_csv(os.path.join(rootPath, 'trainedModel', saveModelName, 'performanceAll.csv'), header=False, index=False, mode='a')


best_dev_epoch = np.argmax(pearson_list_dev)
print("Epoch %d got best Pearson's correlation on dev set: %.4f" % (best_dev_epoch + 1, pearson_list_dev[best_dev_epoch]))
print("Epoch %d got Spearman's correlation on dev set: %.4f" % (best_dev_epoch + 1, spearman_list_dev[best_dev_epoch]))
print("Epoch %d got RMSE on dev set: %.4f" % (best_dev_epoch + 1, rmse_list_dev[best_dev_epoch]))
print("Epoch %d got P@100 POS and NEG on dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_dev[best_dev_epoch][-1][0],
                                                                  precisionk_list_dev[best_dev_epoch][-1][1]))

print("Epoch %d got Pearson's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, pearson_list_test[best_dev_epoch]))
print("Epoch %d got Spearman's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, spearman_list_test[best_dev_epoch]))
print("Epoch %d got RMSE on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, rmse_list_test[best_dev_epoch]))
print("Epoch %d got P@100 POS and NEG on test set w.r.t dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_test[best_dev_epoch][-1][0],
                                                                  precisionk_list_test[best_dev_epoch][-1][1]))

best_test_epoch = np.argmax(pearson_list_test)
print("Epoch %d got best Pearson's correlation on test set: %.4f" % (best_test_epoch + 1, pearson_list_test[best_test_epoch]))
print("Epoch %d got Spearman's correlation on test set: %.4f" % (best_test_epoch + 1, spearman_list_test[best_test_epoch]))
print("Epoch %d got RMSE on test set: %.4f" % (best_test_epoch + 1, rmse_list_test[best_test_epoch]))
print("Epoch %d got P@100 POS and NEG on test set: %.4f, %.4f" % (best_test_epoch + 1,
                                                                  precisionk_list_test[best_test_epoch][-1][0],
                                                                  precisionk_list_test[best_test_epoch][-1][1]))

end_time = datetime.now()
# if(max_epoch >= 100):
#     torch.save(model, os.path.join(rootPath, re.sub('\D', '_', re.sub('\\..*', '', str(end_time))) + '.pth'))


# Iteration 180:
# c:\PharosiBio\Codes\pythonCodes\DeepCE-master\DeepCE\models\neural_fingerprint.py:58: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
#   atom_activations = F.softmax(linear(node_repr))
# Train loss:
# 2.039181747282649
# Dev loss:
# 3.031677464231886
# RMSE: 1.7412
# Pearson's correlation: 0.4212
# Spearman's correlation: 0.4195
# Precision@10 Positive: 0.3818
# Precision@10 Negative: 0.4039
# Precision@20 Positive: 0.3442
# Precision@20 Negative: 0.3719
# Precision@50 Positive: 0.2952
# Precision@50 Negative: 0.3231
# Precision@100 Positive: 0.2536
# Precision@100 Negative: 0.2848
# c:\PharosiBio\Codes\pythonCodes\DeepCE-master\DeepCE\models\neural_fingerprint.py:58: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
#   atom_activations = F.softmax(linear(node_repr))
# Test loss:
# 3.1603305268320168
# RMSE: 1.7777
# Pearson's correlation: 0.4284
# Spearman's correlation: 0.4268
# Precision@10 Positive: 0.3869
# Precision@10 Negative: 0.4083
# Precision@20 Positive: 0.3610
# Precision@20 Negative: 0.3801
# Precision@50 Positive: 0.3032
# Precision@50 Negative: 0.3340
# Precision@100 Positive: 0.2543
# Precision@100 Negative: 0.2873
# Epoch 143 got best Pearson's correlation on dev set: 0.4417
# Epoch 143 got Spearman's correlation on dev set: 0.4401
# Epoch 143 got RMSE on dev set: 1.7234
# Epoch 143 got P@100 POS and NEG on dev set: 0.2483, 0.2897
# Epoch 143 got Pearson's correlation on test set w.r.t dev set: 0.4360
# Epoch 143 got Spearman's correlation on test set w.r.t dev set: 0.4347
# Epoch 143 got RMSE on test set w.r.t dev set: 1.7798
# Epoch 143 got P@100 POS and NEG on test set w.r.t dev set: 0.2493, 0.2867
# Epoch 149 got best Pearson's correlation on test set: 0.4481
# Epoch 149 got Spearman's correlation on test set: 0.4467
# Epoch 149 got RMSE on test set: 1.7578
# Epoch 149 got P@100 POS and NEG on test set: 0.2710, 0.2969

print(end_time - start_time)
print('Done')
