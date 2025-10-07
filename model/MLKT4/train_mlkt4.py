import sys
import pandas as pd
import argparse
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import numpy as np
import torch
from pytorchtools import EarlyStopping
from mlkt_utils import generate_time_intervals_and_write_back
import tqdm
import torch.optim as optim
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from gen_data import KTDataset
from smx import StudentKnowledgeModel
# from smx_wo_logic import StudentKnowledgeModel
from eval_mlkt import lossFunc,performance
from torch.utils.data import DataLoader

seed = 42
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Script to train MLKT4')
# train params
parser.add_argument('--max_iter', type=int, default=500, help='number of iterations')
parser.add_argument('--seed', type=int, default=224, help='default seed')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--max_length', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=12)

# dataset params
parser.add_argument('--data_dir', type=str, default='Code-S', choices=['Code-S', 'Code-F','BePKT'])
parser.add_argument('--model', type=str, default='MLKT4')

# model params
parser.add_argument('--device', type=str, default=device, help='device')
parser.add_argument('--emb_size', type=int, default=256)


params = parser.parse_args()
if params.data_dir in ['Code-F', 'Code-S']:
    NUM_QUESTIONS = 50
    NUM_SKILLS = 18
elif params.data_dir in ['BePKT']:
    NUM_QUESTIONS = 553
    NUM_SKILLS = 207

generate_time_intervals_and_write_back(params.data_dir)
train_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Train')
valid_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Valid')
test_dataset = KTDataset(params, NUM_QUESTIONS, params.batch_size, params.max_length, mode='Test')

train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

# Record the training process
save_log = './log'
if not os.path.exists(save_log):
    os.makedirs(save_log)
log_file = os.path.join(
    save_log +'/wo_ReasonLoss{}_d{}_{}_test_result.txt'.format(params.model,params.emb_size, params.data_dir))
log = open(log_file, 'w')

rmse_test_list = []
mae_test_list = []
now = time.time()
for fold_idx in range(5):
    print(f'current fold is {fold_idx}')
    early_stopping = EarlyStopping(patience=10, verbose=True)

    save_model_file = os.path.join('./fold{}/{}/dim{}S31R3-True'.format(fold_idx, params.data_dir,params.emb_size))

    if not os.path.exists(save_model_file):
        os.makedirs(save_model_file)

    model = StudentKnowledgeModel(NUM_QUESTIONS, NUM_SKILLS, params.emb_size, 8,params.device).to(params.device)

    model_adam = optim.Adam(model.parameters(), lr=params.lr)
    loss_func = lossFunc(params.device)
    # model_path = os.path.join(save_model_file, 'kt_model_best.pt')
    # if os.path.exists(model_path):
    #    model.load_state_dict(torch.load(model_path, weights_only=True))
    #    model.eval()
    #    all_preds = []
    #    all_labels = []
    #    with torch.no_grad():
    #        for batch in tqdm.tqdm(test_dataloader, desc='Testing'):
    #            problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor = batch
    #            pred,_,_,_,_ = model(problem_tensor, skills_tensor, label_tensor, reason_tensor, time_tensor[:, 1:])
    #            # print(time_tensor.max())
    #            # print(time_tensor.min())
    #            all_preds.append(pred[:, 1:])
    #            all_labels.append(note_tensor[:, 1:])
    #            loss = loss_func(pred[:, 1:], label_tensor[:, 1:], note_tensor[:, 1:])
    #
    #        all_preds = torch.cat(all_preds, dim=0)
    #        all_labels = torch.cat(all_labels, dim=0)
    #        rmse, mae = performance(all_labels, all_preds)
    #        rmse_test_list.append(rmse)
    #        mae_test_list.append(mae)
    #        print('test_rmse:', rmse, 'test_mae:', mae)
    #        print('test_rmse:', rmse, 'test_mae:', mae, file=log)
    #
    #        continue



    for epoch in range(params.max_iter):
        print(f'current epoch is {epoch+1}')
        model.train()
        for batch in tqdm.tqdm(train_dataloader, desc='Training'):
            problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor = batch

            # Model now returns predictions and regularization loss
            pred_score,pred_reason_logic,pred_reason_concept,logic_weight,concept_weight = model(problem_tensor, skills_tensor, label_tensor, reason_tensor, time_tensor[:, 1:])
            # print("pred min:", pred.min().item(), "max:", pred.max().item(), "isnan:", torch.isnan(pred).any().item(),
            #       "isinf:", torch.isinf(pred).any().item())
            # print(pred_score.max())
            # print(pred_score.min())
            pred_loss = loss_func(pred_score[:, 1:], label_tensor[:, 1:], note_tensor[:, 1:])

            # Combine prediction loss and regularization loss
            total_loss = pred_loss

            model_adam.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            model_adam.step()

        model.eval()
        all_preds = []
        all_labels = []
        loss_list = torch.Tensor([]).to(params.device)
        with torch.no_grad():
            for batch in tqdm.tqdm(valid_dataloader, desc='Validating'):
                problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor = batch
                # pred, _ = model(problem_tensor, skills_tensor, label_tensor, reason_tensor, time_tensor[:, 1:])
                pred,_,_ ,_,_= model(problem_tensor, skills_tensor, label_tensor, reason_tensor, time_tensor[:, 1:])
                all_preds.append(pred[:, 1:])
                all_labels.append(note_tensor[:, 1:])
                loss = loss_func(pred[:, 1:], label_tensor[:, 1:], note_tensor[:, 1:])
                loss_list = torch.cat((loss_list, torch.tensor([loss], device=params.device)), dim=0)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        valid_rmse, valid_mae = performance(all_labels, all_preds)

        print('valid_rmse:', valid_rmse, 'valid_mae:', valid_mae)

        early_stopping(loss_list.mean(), model,
                       save_path=os.path.join(save_model_file, 'kt_model_best.pt'))
        if early_stopping.early_stop:
            print("Early stopping")
            break

    load_model_path = os.path.join(save_model_file, 'kt_model_best.pt')
    model.load_state_dict(torch.load(load_model_path, weights_only=True))

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, desc='Testing'):
            problem_tensor, skills_tensor, label_tensor, note_tensor, reason_tensor, time_tensor = batch
            pred,_,_,_,_ = model(problem_tensor, skills_tensor, label_tensor, reason_tensor, time_tensor[:, 1:])
            all_preds.append(pred[:, 1:])
            all_labels.append(note_tensor[:, 1:])
            loss = loss_func(pred[:, 1:], label_tensor[:, 1:], note_tensor[:, 1:])

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        rmse, mae = performance(all_labels, all_preds)
        rmse_test_list.append(rmse)
        mae_test_list.append(mae)
        print('test_rmse:', rmse, 'test_mae:', mae)
        print('test_rmse:', rmse, 'test_mae:', mae,file=log)

print('average test rmse:', np.round(np.mean(rmse_test_list), decimals=4), u'\u00B1',
          np.round(np.std(rmse_test_list), decimals=4))
print('average test rmse:', np.round(np.mean(rmse_test_list), decimals=4), u'\u00B1',
          np.round(np.std(rmse_test_list), decimals=4),file=log)
print('average test mae:', np.round(np.mean(mae_test_list), decimals=4), u'\u00B1',
          np.round(np.std(mae_test_list), decimals=4))
print('average test mae:', np.round(np.mean(mae_test_list), decimals=4), u'\u00B1',
          np.round(np.std(mae_test_list), decimals=4),file=log)

end = time.time()
print('total running time:{} min'.format((end - now) / 60))
print('total running time:{} min'.format((end - now) / 60),file=log)
log.close()
print('此为DK3.0')
