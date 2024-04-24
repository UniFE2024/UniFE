import torch
import random


class Server:
    def __init__(self, model, nentity, client_entities_dict, client_entities_mapping, ent_dim, rel_dim, score_func, device):
        self.model = model.to(device)
        self.model_params = {key: value for key, value in self.model.named_parameters()}
        self.nentity = nentity
        self.client_entities_mapping = client_entities_mapping
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.client_entities_dict = client_entities_dict
        self.global_entity_embedding = torch.zeros(self.nentity, self.ent_dim).to(device)
        torch.nn.init.xavier_normal_(self.global_entity_embedding)
        self.device = device

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_struct_params(self, selected_clients):
        total_basis = 0
        total_sr = 0
        for client in selected_clients:
            total_basis += client.train_size['basis_size']
            total_sr += client.train_size['sr_size']
        for k in self.model_params.keys():
            if 'degree_weights_s' in k:
                tmp = []
                for client in selected_clients:
                    tmp.append(torch.mul(client.model_params[k].data, client.train_size['basis_size'].unsqueeze(1)))
                data_agg = torch.div(torch.sum(torch.stack(tmp), dim=0), total_basis.unsqueeze(1))
                self.model_params[k].data = data_agg.clone()
            elif '_agg' in k:
                tmp = []
                for client in selected_clients:
                    tmp.append(torch.mul(client.model_params[k].data, client.train_size['sr_size']))
                data_agg = torch.div(torch.sum(torch.stack(tmp), dim=0), total_sr)
                self.model_params[k].data = data_agg.clone()

    def assign_embeddings(self):
        entity_embedding_dict = dict()
        for client_seq in self.client_entities_mapping.keys():
            client_embedding = self.global_entity_embedding[self.client_entities_mapping[client_seq]].detach().clone()
            entity_embedding_dict[client_seq] = client_embedding
        return entity_embedding_dict

    def aggregate_feat_embeddings(self, entity_embedding_dict):
        later_global_embedding = torch.zeros(self.nentity, self.ent_dim)
        weight = torch.zeros(self.nentity)
        for client_seq in entity_embedding_dict.keys():
            weight[self.client_entities_mapping[client_seq]] += 1
            later_global_embedding[self.client_entities_mapping[client_seq]] += entity_embedding_dict[
                client_seq].cpu().detach()
        standard = torch.ones(weight.shape)
        weight = torch.where(weight > 0, weight, standard)
        weight = weight.view(weight.shape[0], 1)
        later_global_embedding /= weight
        self.global_entity_embedding = later_global_embedding.to(self.device)
