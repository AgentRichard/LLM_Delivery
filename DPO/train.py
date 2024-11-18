from model_train import GQA
from model_ref import GQA
from data import Data
import torch
from torch.optim.adam import Adam
from dpoloss import DPOLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

config = {
    "vinput" : 240,
    "hidden" : 728,
    "qheads" : 12,
    "kvheads": 2,
    "rope_len" : 5000,
    "voc_size" : 30000,
    "cache_max_batch" : None,
    "cache_max_seq" : None,
    "num_layers" : 3
}

class Train:
    def __init__(self, data_dir, seq_len):
        self._train_net = GQA(**config).cuda()
        self._ref_net = GQA(**config).cuda()
        ### 仅有一个参与训练，因此仅需要一个优化器
        self._opt = Adam(self._train_net.parameters(), lr=0.001, betas=(0.85,0.95), weight_decay=0.1)
        self._train_dataset = Data(data_dir, seq_len)
        self._train_data = DataLoader(self._train_dataset, 10, shuffle=True)
        self._lossfn = DPOLoss()
        self._log = SummaryWriter("/root/project/My_projects/Transformer/GQA_RLHF/runs")
        
    def __call__(self):
        for _epoch in range(2):
            self._train_net.train()
            self._ref_net.eval()
            
            for _i, (chosen_input_ids, chosen_label_ids, rejected_input_ids, rejected_label_ids) in enumerate(self._train_data):
                chosen_input_ids = chosen_input_ids.to(device="cuda", dtype=torch.long)
                rejected_input_ids = rejected_input_ids.to(device="cuda", dtype=torch.long)
                
                chosen_label_ids = chosen_label_ids.to(device="cuda", dtype=torch.long)
                rejected_label_ids = rejected_label_ids.to(device="cuda", dtype=torch.long)
                
                chosen_logit = self._train_net(chosen_input_ids)
                reject_logit = self._train_net(rejected_input_ids)
                
                with torch.no_grad():
                    ref_chosen_logit = self._ref_net(chosen_input_ids)
                    ref_reject_logit = self._ref_net(rejected_input_ids)
                
                chosen_logit = chosen_logit.reshape(-1, 30000)
                reject_logit = reject_logit.reshape(-1, 30000)
                ref_chosen_logit = ref_chosen_logit.reshape(-1, 30000)
                ref_reject_logit = ref_reject_logit.reshape(-1, 30000)
                
                chosen_label_ids = chosen_label_ids.reshape(-1)
                rejected_label_ids = rejected_label_ids.reshape(-1)
                
                _loss = self._lossfn(chosen_logit, ref_chosen_logit, reject_logit, ref_reject_logit, chosen_label_ids, rejected_label_ids)
                
                self._opt.zero_grad()
                _loss.backward()
                self._opt.step()
                if _i % 5 == 0:
                    self._log.add_scalar("Loss", _loss.detach()*100, global_step=_i)
                    print("loss", _loss.detach()*100, _epoch)

if __name__ == "__main__":
    train = Train("/root/project/My_projects/Transformer/GQA_RLHF/datas/data/Chinese_dpo_pairs.bin", 250)
    train()
                
                
                