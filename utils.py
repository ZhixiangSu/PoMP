import numpy as np
from sklearn.metrics import average_precision_score

from torch.utils.data import Dataset, DataLoader

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def cal_metrics(category_predictions,category_labels,disease_predictions,disease_labels):
    category_predictions=np.array(category_predictions)
    category_labels=np.array(category_labels)
    sorted_index=np.argsort(-category_predictions,axis=1)
    pos=np.array([np.where(sorted_index[i]==category_labels[i]) for i in range(len(category_labels))]).flatten()
    hit1 = len(pos[pos<1])/len(pos)
    hit3 = len(pos[pos < 3])/len(pos)
    hit10 = len(pos[pos < 10])/len(pos)

    AUC_PR=average_precision_score(one_hot(category_labels,category_predictions.shape[1]).reshape(-1),category_predictions.reshape(-1))

    # MR=np.average(pos+1)
    # MRR=np.average(1/(pos+1))

    category_mask = pos==0

    disease_predictions = np.array(disease_predictions)
    disease_labels = np.array(disease_labels)

    mask_idxs=np.stack([np.arange(len(disease_labels)),disease_labels],axis=1)[category_mask]
    disease_predictions[mask_idxs[:,0],mask_idxs[:,1]] = 0
    disease_sorted_index = np.argsort(-disease_predictions, axis=1)
    disease_pos = np.array([np.where(disease_sorted_index[i] == disease_labels[i]) for i in range(len(disease_labels))]).flatten()
    disease_hit1 = len(disease_pos[disease_pos<1]) / len(disease_pos)
    disease_hit3 = len(disease_pos[disease_pos<3]) / len(disease_pos)
    disease_hit10 = len(disease_pos[disease_pos<10]) / len(disease_pos)

    # disease_hit1 = len(disease_pos[np.logical_and(disease_pos<1,category_mask)]) / len(disease_pos)
    # disease_hit3 = len(disease_pos[np.logical_and(disease_pos<3,category_mask)]) / len(disease_pos)
    # disease_hit10 = len(disease_pos[np.logical_and(disease_pos<10,category_mask)]) / len(disease_pos)
    disease_AUC_PR=average_precision_score(one_hot(disease_labels,disease_predictions.shape[1]).reshape(-1),disease_predictions.reshape(-1))
    # disease_MR = np.average(disease_pos + 1)
    # disease_MRR = np.average(1 / (disease_pos + 1))

    return [hit1,hit3,hit10,AUC_PR],[disease_hit1,disease_hit3,disease_hit10,disease_AUC_PR]
class PandasDataset(Dataset):
    def __init__(self, dataframe,category_dict,disease_dict):
        self.dataframe = dataframe
        self.category_dict=category_dict
        self.disease_dict = disease_dict

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        gender=row['gender']
        pregnancy = row['pregnancy situation']
        profile=row[['age', 'height', 'weight']].to_list()
        description=row['description']
        category=self.category_dict[row['category']]
        disease=self.disease_dict[category][row['disease']]

        return [gender,pregnancy,profile,description,category,disease]

    def __len__(self):
        return len(self.dataframe)
