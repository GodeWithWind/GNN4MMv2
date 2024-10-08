from data.dataset import HetDataset
from models.gnn import GATReal,HetGATV2, CNN,MLP,GCN,GAT,GATV3

model_dict = { "GATReal":GATReal ,"HetGATV2":HetGATV2,"CNN":CNN,"MLP":MLP,"GCN":GCN,"GAT":GAT,"GATV3":GATV3}


def build_model(args):
    if args.model not in model_dict:
        raise ValueError('Invalid model name!')
    if args.model == "HetGATV2":
        dataset = HetDataset(args.train_path)
        data = dataset[0]
        model = HetGATV2(data.metadata(), args)
    else:
        model_class = model_dict[args.model]
        model = model_class(args)
    return model
