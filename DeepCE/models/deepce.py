import torch, gc, time
import torch.nn as nn
from .neural_fingerprint import NeuralFingerprint
from .drug_gene_attention import DrugGeneAttention
from .ltr_loss import *
from utils.data_utils import *

class DeepCE(nn.Module):
    def __init__(self, drug_input_dim, drug_emb_dim, conv_size, degree, attentionType, gene_input_dim, gene_emb_dim, num_gene, cell_input_dim, cell_emb_dim, 
                 n_layers, n_heads, pf_dim, hid_dim, dropout, loss_type, device, initializer=None, 
                 pert_type_input_dim=None, pert_idose_input_dim=None,
                 pert_type_emb_dim=None, cell_id_emb_dim=None, pert_idose_emb_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False):
        super(DeepCE, self).__init__()
        self.device = device
        # assert drug_emb_dim == gene_emb_dim, 'Embedding size mismatch'
        # assert cell_emb_dim == gene_emb_dim, 'Embedding size mismatch'
        self.use_pert_type = use_pert_type
        self.use_cell_id = use_cell_id
        self.use_pert_idose = use_pert_idose
        self.drug_emb_dim = drug_emb_dim
        self.gene_emb_dim = gene_emb_dim
        self.cell_emb_dim = cell_emb_dim
        self.drug_fp = NeuralFingerprint(drug_input_dim['atom'], drug_input_dim['bond'], conv_size, drug_emb_dim,
                                         degree, device)
        self.gene_embed = nn.Linear(gene_input_dim, gene_emb_dim)
        self.cell_embed = nn.Linear(cell_input_dim, cell_emb_dim)
        # self.linear_dim = self.drug_emb_dim + self.gene_emb_dim
        self.drug_gene_attn = DrugGeneAttention(gene_emb_dim, gene_emb_dim, n_layers, n_heads, pf_dim,
                                        dropout=dropout, device=device)
        self.cell_gene_attn = DrugGeneAttention(gene_emb_dim, gene_emb_dim, n_layers, n_heads, pf_dim,
                                        dropout=dropout, device=device)
        self.drug_cell_attn = DrugGeneAttention(gene_emb_dim, gene_emb_dim, n_layers, n_heads, pf_dim,
                                                dropout=dropout, device=device)
        self.linear_dim = 0
        for typeI in attentionType:
            self.linear_dim = self.linear_dim + self.drug_emb_dim
        if self.use_pert_type:
            self.pert_type_embed = nn.Linear(pert_type_input_dim, pert_type_emb_dim)
            self.linear_dim += pert_type_emb_dim
        if self.use_pert_idose:
            self.pert_idose_embed = nn.Linear(pert_idose_input_dim, pert_idose_emb_dim)
            self.linear_dim += pert_idose_emb_dim
        self.linear_1 = nn.Linear(self.linear_dim, hid_dim)
        self.linear_2 = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.num_gene = num_gene
        self.loss_type = loss_type
        self.initializer = initializer
        
        self.init_weights()
        self.attentionType = attentionType

    def init_weights(self):
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            if ('drug_gene_attn' not in name) and ('cell_gene_attn' not in name) and ('drug_cell_attn' not in name):
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 0.)
                else:
                    self.initializer(parameter)

    def forward(self, drugIndex, input_drug, input_gene, input_pert_type, input_cell_id, input_pert_idose):
        # # input_drug = {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}
        # # gene_embed = [num_gene * gene_emb_dim]
        if(torch.cuda.device_count() > 1):
            input_drug = input_drug[drugIndex.cpu().numpy()]
            input_drug = convert_smile_to_feature(input_drug, self.device)
        else:
            input_drug = convert_smile_to_feature(input_drug, self.device)
        mask = create_mask_feature(input_drug, self.device)
        num_batch = input_drug['molecules'].batch_size
        input_drug['atom'] = input_drug['atom'].to(self.device)
        # print(torch.cuda.memory_allocated())
        input_drug['bond'] = input_drug['bond'].to(self.device)
        drug_atom_embed = self.drug_fp(input_drug)

        # batch_idx = input_drug['molecules'].get_neighbor_idx_by_batch('atom')
        # molecule_length = [len(idx) for idx in batch_idx]
        # max_length = max(molecule_length)
        # num_atom = sum(molecule_length)
        # drug_atom_embed = self.drug_fp(input_drug, num_batch, batch_idx, num_atom, max_length)
        
        # gc.collect()
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())
        # drug_atom_embed = [batch * num_node * drug_emb_dim]
        if(sum([('Drug' in teypI) for teypI in self.attentionType]) == 0):
            drug_embed = None
        else:
            drug_embed = torch.sum(drug_atom_embed, dim=1)
            # drug_embed = [batch * drug_emb_dim]
            drug_embed = drug_embed.unsqueeze(1)
            # drug_embed = [batch * 1 *drug_emb_dim]
            drug_embed = drug_embed.repeat(1, self.num_gene, 1)
            # drug_embed = [batch * num_gene * drug_emb_dim]
        if(sum([('Cell' in teypI) for teypI in self.attentionType]) == 0):
            cell_embed = None
        else:
            cell_embed = self.cell_embed(input_cell_id.to(self.device))
            cell_embed = cell_embed.unsqueeze(1)
            cell_embed = cell_embed.repeat(1, self.num_gene, 1)
        if(sum([('Gene' in teypI) for teypI in self.attentionType]) == 0):
            gene_embed = None
        else:
            gene_embed = self.gene_embed(torch.from_numpy(input_gene).to(self.device))
            # gene_embed = [num_gene * gene_emb_dim]
            gene_embed = gene_embed.unsqueeze(0)
            # gene_embed = [1 * num_gene * gene_emb_dim]
            gene_embed = gene_embed.repeat(num_batch, 1, 1)
            # gene_embed = [batch * num_gene * gene_emb_dim]
        if('Drug_Gene' in self.attentionType) or ('Drug_Cell' in self.attentionType):
            mask = mask.to(self.device)
        else:
            mask = None
        if('Drug_Gene' in self.attentionType):
            drug_gene_embed, _ = self.drug_gene_attn(gene_embed, drug_atom_embed, None, mask)
        else:
            drug_gene_embed = None
        if('Drug_Cell' in self.attentionType):
            drug_cell_embed, _ = self.drug_cell_attn(cell_embed, drug_atom_embed, None, mask)
        else:
            drug_cell_embed = None
        if('Cell_Gene' in self.attentionType):
            cell_gene_embed, _ = self.drug_cell_attn(gene_embed, cell_embed, None, None)
        else:
            cell_gene_embed = None

        if(not 'Drug' in self.attentionType):
            drug_embed = None
        if(not 'Cell' in self.attentionType):
            cell_embed = None
        if(not 'Gene' in self.attentionType):
            gene_embed = None
            
        embed_list = [x for x in [drug_gene_embed, drug_cell_embed, cell_gene_embed, drug_embed, cell_embed, gene_embed] if x is not None]
        final_embed = torch.cat(embed_list, dim=2)
        # drug_gene_embed = [batch * num_gene * gene_emb_dim]
        # drug_gene_embed = [batch * num_gene * (drug_emb_dim + gene_emb_dim)]
        if self.use_pert_type:
            pert_type_embed = self.pert_type_embed(input_pert_type.to(self.device))
            # pert_type_embed = [batch * pert_type_emb_dim]
            pert_type_embed = pert_type_embed.unsqueeze(1)
            # pert_type_embed = [batch * 1 * pert_type_emb_dim]
            pert_type_embed = pert_type_embed.repeat(1, self.num_gene, 1)
            # pert_type_embed = [batch * num_gene * pert_type_emb_dim]
            final_embed = torch.cat((final_embed, pert_type_embed), dim=2)
        # if self.use_cell_id:
        #     cell_id_embed = self.cell_id_embed(input_cell_id)
        #     # cell_id_embed = [batch * cell_id_emb_dim]
        #     cell_id_embed = cell_id_embed.unsqueeze(1)
        #     # cell_id_embed = [batch * 1 * cell_id_emb_dim]
        #     cell_id_embed = cell_id_embed.repeat(1, self.num_gene, 1)
        #     # cell_id_embed = [batch * num_gene * cell_id_emb_dim]
        #     drug_gene_embed = torch.cat((drug_gene_embed, cell_id_embed), dim=2)
        if self.use_pert_idose:
            pert_idose_embed = self.pert_idose_embed(input_pert_idose.to(self.device))
            # pert_idose_embed = [batch * pert_idose_emb_dim]
            pert_idose_embed = pert_idose_embed.unsqueeze(1)
            # pert_idose_embed = [batch * 1 * pert_idose_emb_dim]
            pert_idose_embed = pert_idose_embed.repeat(1, self.num_gene, 1)
            # pert_idose_embed = [batch * num_gene * pert_idose_emb_dim]
            final_embed = torch.cat((final_embed, pert_idose_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * (drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        final_embed = self.relu(final_embed)
        # drug_gene_embed = [batch * num_gene * (drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        out = self.linear_1(final_embed)
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        out = self.linear_2(out)
        # out = [batch * num_gene * 1]
        out = out.squeeze(2)
        # out = [batch * num_gene]
        return out

    def loss(self, label, predict):
        label = label.to(self.device)
        if self.loss_type == 'point_wise_mse':
            loss = point_wise_mse(label, predict)
        elif self.loss_type == 'pair_wise_ranknet':
            loss = pair_wise_ranknet(label, predict, self.device)
        elif self.loss_type == 'list_wise_listnet':
            loss = list_wise_listnet(label, predict)
        elif self.loss_type == 'list_wise_listmle':
            loss = list_wise_listmle(label, predict, self.device)
        elif self.loss_type == 'list_wise_rankcosine':
            loss = list_wise_rankcosine(label, predict)
        elif self.loss_type == 'list_wise_ndcg':
            loss = list_wise_ndcg(label, predict)
        elif self.loss_type == 'pearson':
            loss = pearson(label, predict)
        elif self.loss_type == 'pearsonJmche':
            loss = pearsonJmche(label, predict)
        elif self.loss_type == 'cosineSimJmche':
            loss = cosineSimJmche(label, predict)
        elif self.loss_type == 'pearsonNcosine':
            loss = pearsonNcosine(label, predict)
        else:
            raise ValueError('Unknown loss: %s' % self.loss_type)
        return loss
