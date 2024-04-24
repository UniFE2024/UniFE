import os
import sys
import torch
from tqdm import tqdm


class Client:
    def __init__(self, args, model, client_id, client_name, train_size, dataLoader, optimizer, kg, bsg):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args
        self.kg = kg
        self.bsg = bsg
        self.model_params = {key: value for key, value in self.model.named_parameters()}

    def download_struct_params(self, server):
        for k in server.model_params:
            if '_agg' in k:
                self.model_params[k].data = server.model_params[k].detach().clone()

    def download_nembds(self, idx, server, entity_embedding_dict):
        for k in server.model_params:
            if 'n_embds' in k:
                self.model_params[k].data = entity_embedding_dict[idx].detach().clone()

    def get_entity_embeddings(self):
        return self.model.feat_and_struct_encoder.n_embds.detach().clone()

    def local_train(self, local_epoch):
        train_loader, val_head_loader, val_tail_loader, test_head_loader, test_tail_loader = self.dataLoader['train'], self.dataLoader['valid_head'], \
            self.dataLoader['valid_tail'], self.dataLoader['test_head'], self.dataLoader['test_tail']
        for epoch in range(1, local_epoch + 1):
            self.model.train()
            total_loss = 0.
            total_num = 0
            for _, batch in tqdm(enumerate(train_loader), total=len(train_loader), file=sys.stdout, leave=False):
                triple, label = batch[0].to(self.args.device), batch[1].to(self.args.device)
                sub, rel, obj = triple[:, 0], triple[:, 1], triple[:, 2]
                if self.args.use_structure:
                    logits = self.model(self.kg, self.bsg, sub, rel)
                else:
                    logits = self.model(self.kg, sub, rel)
                tr_loss = torch.nn.BCELoss()(logits, label)
                self.optimizer.zero_grad()
                tr_loss.backward()
                self.optimizer.step()
                total_num += batch[0].shape[0]
                total_loss += tr_loss.item() * batch[0].shape[0]
        return

    def evaluate_unife(self, eval_mode):
        if eval_mode == 'valid':
            dataloader_head = self.dataLoader['valid_head']
            dataloader_tail = self.dataLoader['valid_tail']
        elif eval_mode == 'test':
            dataloader_head = self.dataLoader['test_head']
            dataloader_tail = self.dataLoader['test_tail']
        else:
            raise ValueError('Not support eval_mode.')
        left_results = self.predict_unife(self.model, self.kg, self.bsg, self.args.use_structure, self.args.device, dataloader_head)
        right_results = self.predict_unife(self.model, self.kg, self.bsg, self.args.use_structure, self.args.device, dataloader_tail)
        results = {}
        count = float(left_results["count"])
        results["left_mr"] = left_results["mr"] / count
        results["left_mrr"] = left_results["mrr"] / count
        results["right_mr"] = right_results["mr"] / count
        results["right_mrr"] = right_results["mrr"] / count
        results["mr"] = (left_results["mr"] + right_results["mr"]) / (2 * count)
        results["mrr"] = (left_results["mrr"] + right_results["mrr"]) / (2 * count)
        results["left_loss"] = left_results["loss"]
        results["right_loss"] = right_results["loss"]
        results["loss"] = (left_results["loss"] + right_results["loss"]) / 2.0
        for k in [1, 3, 10]:
            results["left_hits@{}".format(k)] = left_results["hits@{}".format(k)] / count
            results["right_hits@{}".format(k)] = right_results["hits@{}".format(k)] / count
            results["hits@{}".format(k)] = (left_results["hits@{}".format(k)] + right_results["hits@{}".format(k)]) / (2 * count)
        main_results = {}
        for key in ['loss', 'mrr', 'hits@10', 'hits@3', 'hits@1', 'mr']:
            main_results[key] = results[key]
        main_results['count'] = 2 * count
        return main_results, results

    def predict_unife(self, model, kg, bsg, use_structure, device, test_loader):
        model.eval()
        total_loss = 0.
        total_num = 0
        with torch.no_grad():
            results = {}
            for step, batch in enumerate(test_loader):
                triple, label = batch[0].to(device), batch[1].to(device)
                sub, rel, obj = triple[:, 0], triple[:, 1], triple[:, 2]
                if use_structure:
                    pred = model(kg, bsg, sub, rel)
                else:
                    pred = model(kg, sub, rel)
                tr_loss = torch.nn.BCELoss()(pred, label)
                total_num += batch[0].shape[0]
                total_loss += tr_loss.item() * batch[0].shape[0]
                b_range = torch.arange(pred.size()[0], device=device)
                target_pred = pred[b_range, obj]
                pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred

                ranks = (1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj])
                ranks = ranks.float()
                results["count"] = torch.numel(ranks) + results.get("count", 0.0)
                results["mr"] = torch.sum(ranks).item() + results.get("mr", 0.0)
                results["mrr"] = torch.sum(1.0 / ranks).item() + results.get("mrr", 0.0)
                for k in [1, 3, 10]:
                    results["hits@{}".format(k)] = torch.numel(ranks[ranks <= (k)]) + results.get("hits@{}".format(k), 0.0)
            results["loss"] = total_loss / total_num
        return results

    def save_model(self, round):
        state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'round': round}
        for filename in os.listdir(self.args.pt_dir):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.args.pt_dir, filename)):
                os.remove(os.path.join(self.args.pt_dir, filename))
        torch.save(state, os.path.join(self.args.pt_dir, self.name + '.' + str(round) + '.pt'))

    def load_model(self, round):
        ckpt = torch.load(os.path.join(self.args.pt_dir, self.name + '.' + str(round) + '.pt'))
        self.model.load_state_dict(ckpt['net'])
