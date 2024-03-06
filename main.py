import os
import argparse
import pandas as pd
import numpy as np
import sys

from tqdm import tqdm
from colorama import Fore

from torch.cuda.amp import GradScaler, autocast
from transformers import set_seed

from torch.optim import lr_scheduler
from collections import defaultdict
from models import PoMP
from utils import PandasDataset,cal_metrics

import torch
from torch.utils.data import DataLoader, random_split

parser = argparse.ArgumentParser(description="Disease Prediction")
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--model_save_dir', type=str, default='results/')
parser.add_argument('--model_load_file', type=str, default='results/best_val.pth')
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--in_dim', type=int, default=3)
parser.add_argument('--attention_dim', type=int, default=64)
parser.add_argument('--gender_dim', type=int, default=32)
parser.add_argument('--pregnancy_dim', type=int, default=32)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
parser.add_argument('--tokenizer', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
parser.add_argument('--hierarchical_loss_weight', type=float, default=0.7)
parser.add_argument('--eval_metric', type=int, default=1)

args = parser.parse_args()
print(args)
set_seed(args.seed)

categories=['chd','cold','depr','diab','lung','pneu']
category_dict={c:i for i,c in enumerate(categories)}
disease_dict={i:{} for i in range(len(categories))}



data=[]

for c in categories:
    with open(os.path.join(args.data_path,f'{c}_inter.csv'),'r',encoding='utf-8') as f:
        d=pd.read_csv(f,sep='\t',dtype={'pregnancy situation':int,'gender': int, 'age': float,'height': float,'weight': float, 'duration of illness':float})
        # df=pd.DataFrame(columns=["chronic disease","surgery history","radiotherapy and chemotherapy history","disease history","medication usage","disease_description",'gender', 'age','height','weight','pregnancy situation','duration of illness','category','disease'])
        # df[["chronic disease","surgery history","radiotherapy and chemotherapy history","disease history","medication usage","disease_description"]]=d[["chronic disease","surgery history","radiotherapy and chemotherapy history","disease history","medication usage","disease"]].replace(np.nan, "")

        df=pd.DataFrame(columns=['description','gender', 'age','height','weight','pregnancy situation','duration of illness','category','disease'])
        df['description']=d['text_all_patient'].replace(np.nan, '')
        df[['age','height','weight','duration of illness']]=d[['age','height','weight','duration of illness']].replace(np.nan, 0.0)
        df['gender']=d['gender'].replace(np.nan, 0)
        df['pregnancy situation'] = d['pregnancy situation'].replace(np.nan, 0)

        df['category']=[c]*d.shape[0]
        df['disease']=d['disease_tag'].replace(np.nan, "")

        disease_dict[category_dict[c]]={d:i for i,d in enumerate(list(set(df['disease'])))}
        data.append(df)
data=pd.concat(data,ignore_index=True)

disease_num=[len(disease_dict[category_dict[categories[i]]]) for i in range(len(categories))]

dataset=PandasDataset(data,category_dict,disease_dict)
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

scaler = GradScaler()


model=PoMP(args.in_dim,args.attention_dim,args.gender_dim,args.pregnancy_dim,args.n_heads,args.dropout,
                len(categories),disease_num,
    tokenizer_name=args.tokenizer,model_name=args.model,device=args.device)

model.to(args.device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = torch.optim.AdamW(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)
scheduler=lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.8)

criterion=torch.nn.CrossEntropyLoss()


def train():
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        # ============================================ TRAINING ============================================================
        print(f"Training epoch {epoch}")
        training_pbar = tqdm(total= train_size,
                             position=0, leave=True,
                             file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET))
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(train_data_loader):
            batch_gender,batch_pregnancy,batch_profile,batch_description,batch_category,batch_disease = batch
            batch_category=batch_category.to(args.device)
            batch_disease=batch_disease.to(args.device)
            optimizer.zero_grad()
            with autocast():
                category_outputs,disease_outputs=model(batch_gender,batch_pregnancy,batch_profile,batch_description)
                disease_outputs=disease_outputs[batch_category,torch.arange(len(batch_category))]
                category_loss = criterion(category_outputs, batch_category)
                disease_loss = criterion(disease_outputs,batch_disease)
                loss=category_loss+args.hierarchical_loss_weight*disease_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
            nb_tr_steps += 1
            training_pbar.update(batch_gender.shape[0])
        training_pbar.close()
        scheduler.step()
        print(f"Learning rate={optimizer.param_groups[0]['lr']}\nTraining loss={tr_loss / nb_tr_steps:.4f}")
        if epoch % 1 == 0:
            valid_acc=validate()
            test()
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                if not os.path.exists(args.model_save_dir):
                    os.makedirs(args.model_save_dir)
                torch.save(model.state_dict(), os.path.join(args.model_save_dir,'best_val.pth'))

    print(f"Best Validation Accuracy: {best_val_acc}")

def validate():
    valid_pbar = tqdm(total=valid_size,
                     position=0, leave=True,
                     file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    model.eval()
    category_metrics = np.array([ 0., 0., 0., 0.])  # MR, MRR, Hit@1, Hit@3, Hit@10
    disease_metrics = np.array([ 0., 0., 0., 0.])  # MR, MRR, Hit@1, Hit@3, Hit@10
    nb_valid_steps = 0
    for batch in valid_data_loader:
        batch_gender, batch_pregnancy, batch_profile, batch_description, batch_category, batch_disease = batch
        optimizer.zero_grad()
        with autocast():
            with torch.no_grad():
                category_outputs,disease_outputs=model(batch_gender,batch_pregnancy,batch_profile,batch_description)
                disease_outputs=disease_outputs[batch_category, torch.arange(len(batch_category))]
        cm,dm = cal_metrics(category_outputs.cpu().numpy(), batch_category.numpy(),disease_outputs.cpu().numpy(), batch_disease.numpy())
        category_metrics+=cm
        disease_metrics+=dm
        nb_valid_steps +=1
        valid_pbar.update(batch_gender.shape[0])
    valid_pbar.close()
    category_metrics = category_metrics / nb_valid_steps
    disease_metrics = disease_metrics / nb_valid_steps
    print("Category:")
    print(
        f" Hit@1: {category_metrics[0]}, Hit@3: {category_metrics[1]}, Hit@10: {category_metrics[2]}, AUC-PR: {category_metrics[3]}")

    print("Disease:")
    print(
        f" Hit@1: {disease_metrics[0]}, Hit@3: {disease_metrics[1]}, Hit@10: {disease_metrics[2]}, AUC-PR: {disease_metrics[3]}")
    return disease_metrics[args.eval_metric]
def test():
    ranking_pbar = tqdm(total=test_size,
                        position=0, leave=True,
                        file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
    model.eval()
    category_metrics=np.array([0.,0.,0., 0.]) # MR, MRR, Hit@1, Hit@3, Hit@10
    disease_metrics = np.array([0., 0., 0., 0.])  # MR, MRR, Hit@1, Hit@3, Hit@10
    nb_ranking_steps = 0
    # scores=[]
    # ranking_positions=[]
    for batch in test_data_loader:
        batch_gender, batch_pregnancy, batch_profile, batch_description, batch_category, batch_disease = batch
        optimizer.zero_grad()

        with autocast():
            with torch.no_grad():
                category_outputs, disease_outputs = model(batch_gender, batch_pregnancy, batch_profile, batch_description)
                disease_outputs = disease_outputs[batch_category, torch.arange(len(batch_category))]
        cm, dm = cal_metrics(category_outputs.cpu().numpy(), batch_category.numpy(), disease_outputs.cpu().numpy(),
                             batch_disease.numpy())
        category_metrics += cm
        disease_metrics += dm

        # batch_scores = np.array(outputs.cpu().numpy())
        # batch_positions = np.argsort(-batch_scores, axis=1)
        # scores.append(batch_scores)
        # ranking_positions.append(batch_positions)
        nb_ranking_steps += 1
        ranking_pbar.update(batch_gender.shape[0])

    category_metrics = category_metrics / nb_ranking_steps
    disease_metrics = disease_metrics / nb_ranking_steps
    ranking_pbar.close()
    print("Category:")
    print(
        f" Hit@1: {category_metrics[0]}, Hit@3: {category_metrics[1]}, Hit@10: {category_metrics[2]}, AUC-PR: {category_metrics[3]}")

    print("Disease:")
    print(
        f" Hit@1: {disease_metrics[0]}, Hit@3: {disease_metrics[1]}, Hit@10: {disease_metrics[2]}, AUC-PR: {disease_metrics[3]}")
    # scores=np.concatenate(scores)
    # ordered_scores=np.array([scores[i][list(ranking_indexes[i])] for i in range(len(scores))])
    # ordered_positions=np.argsort(-ordered_scores, axis=1)
    # false_predicted_triplets=[]
    # false_positions=[]
    #
    # for i in range(len(ordered_positions)):
    #     if ordered_positions[i][0]!=0:
    #         false_predicted_triplets.append(ranking_triplets[i][ranking_labels[i]])
    #         false_positions.append(ordered_positions[i])
    # false_positions=np.array(false_positions)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # np.savetxt(os.path.join(output_dir,"false_positions.txt"), false_positions,fmt="%d", delimiter="\t")
    #
    # with open(os.path.join(output_dir,"false_triplets.txt"),'w') as f:
    #     for triplet in false_predicted_triplets:
    #         f.write("\t".join(triplet)+"\n")
    #     f.close()
    # np.savetxt(os.path.join(output_dir,"indexes.txt"),ordered_positions,fmt="%d", delimiter="\t")
    # np.savetxt(os.path.join(output_dir,"scores.txt"), ordered_scores,fmt="%.5f", delimiter="\t")
try:
    train()
except KeyboardInterrupt:
    print("Receive keyboard interrupt, start testing:")
    model.load_state_dict(torch.load(args.model_load_file, map_location=args.device))
    test()
# model=torch.nn.DataParallel(model,device_ids=[0,1])
model.load_state_dict(torch.load(args.model_load_file, map_location=args.device))
test()