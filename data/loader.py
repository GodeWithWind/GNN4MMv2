from torch_geometric.data import DataLoader as pygDL
from torch.utils.data import  DataLoader as normDL
from data.dataset import HetDataset , CNNDataset , MLPDataset , HomDataset

dataset_dict = {'GATReal': HomDataset,'HetGATV2': HetDataset,'CNN': CNNDataset,'MLP': MLPDataset,'GCN': HomDataset,'GAT': HomDataset,'GATV3': HomDataset}


def get_train_loader(train_path, user_num, batch_size, model, num_workers=4, **kwargs):
    dataset_class = dataset_dict[model]
    dataset = dataset_class(train_path)

    if model == "CNN" or model == "MLP":
        return normDL(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,drop_last=True)
    else:
        return pygDL(dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,drop_last=True)



def get_val_loader(val_path, user_num, batch_size, model, num_workers=4, **kwargs):
    dataset_class = dataset_dict[model]
    dataset = dataset_class(val_path)
    # 返回loader
    if model == "CNN" or model == "MLP":
        return normDL(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,drop_last=True)
    else:
        return pygDL(dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,drop_last=True)


def get_test_loader(test_path, user_num, batch_size, model, num_workers=4, **kwargs):
    dataset_class = dataset_dict[model]
    dataset = dataset_class(test_path)
    # 返回loader
    if model == "CNN" or model == "MLP":
        return normDL(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,drop_last=True)
    else:
        return pygDL(dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,drop_last=True)
