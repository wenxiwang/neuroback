import torch
import torch.nn as nn
from torch.nn import NLLLoss, CrossEntropyLoss, BCELoss

import torch_geometric
from torch_geometric.loader import DataLoader

from data import *
from gt_model import GTModel

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

import texttable as tt

from tqdm import tqdm
import time
import sys
import random
import numpy as np
from multiprocessing import Pool


# hyperparameter setting
hyper_params = {}
if sys.argv[1] == "pretrain":
	hyper_params["pretrain"] = True
elif sys.argv[1] == "finetune":
	hyper_params["pretrain"] = False
else:
	print("argv error! should be either pretrain or finetune!")
	exit(1)

if hyper_params["pretrain"]:
	hyper_params["seed"] = 77
	hyper_params["lr"] = 1e-4
	hyper_params["epoch_num"] = 40
	hyper_params["batch_size"] = 2
	hyper_params["log_dir"] = "./log/pretrain"
	hyper_params["checkpoint_path"] = None
	hyper_params["dataset_path"] = "./data/pt/pretrain"

else:
	hyper_params["seed"] = 77
	hyper_params["lr"] = 1e-4
	hyper_params["epoch_num"] = 60
	hyper_params["batch_size"] = 1
	hyper_params["log_dir"] = "./log/finetune" 
	hyper_params["checkpoint_path"] = "./models/pretrain/pretrain-best.ptg"
	hyper_params["dataset_path"] = "./data/pt/finetune"

# create log folder and model folder
if not os.path.isdir(hyper_params["log_dir"]):
	os.makedirs(hyper_params["log_dir"])

if hyper_params["pretrain"] and not os.path.isdir("./models/pretrain"):
	os.makedirs("./models/pretrain")

if not hyper_params["pretrain"] and not os.path.isdir("./models/finetune"):
	os.makedirs("./models/finetune")

# set up training and validation sets
torch.manual_seed(hyper_params["seed"])
dataset_train = MyOwnDataset(root=hyper_params["dataset_path"])
dataset_train = dataset_train.shuffle()
dataset_vld = MyOwnDataset(root='./data/pt/validation')

train_loader = DataLoader(dataset_train, batch_size=hyper_params["batch_size"], shuffle=True, pin_memory=True, num_workers=2)
vld_loader = DataLoader(dataset_vld, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

# load model and optimizer weights
model = GTModel(3, 3).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["lr"])

if hyper_params["checkpoint_path"] is not None and os.path.isfile(hyper_params["checkpoint_path"]):
	checkpoint = torch.load(hyper_params["checkpoint_path"])
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# train and evaluate
def train(log_file):
	global model
	global optimizer

	model.train()
	total_loss = 0
	total_var_cnt = 0

	all_target = []
	all_pred_class = []

	cnt = 0

	fit_idx_lst = []
	with tqdm(total=len(dataset_train)) as pbar:
		for data in train_loader:
			if data.y == None:
				log_file.write("no data.y, ignore\n")
				continue

			loss = None
			pred = None
			y01 = None

			try:
				model = model.cuda()
				data = data.cuda()

				y01_indices = (data.y != 2).nonzero(as_tuple=True)
				y01 = data.y[y01_indices].float()

				u0 = torch.sum((y01 == 0).int()).cpu()
				u1 = torch.sum((y01 == 1).int()).cpu()
				u01 = u0 + u1
				assert(u01 == y01.shape[0])
				weight = torch.zeros(u01).cuda()
				weight[y01 == 0] = u01 / (2 * (u0 + 1))
				weight[y01 == 1] = u01 / (2 * (u1 + 1))
				crit = BCELoss(weight=weight.view(-1, 1))
				
				optimizer.zero_grad(set_to_none=True)

				pred = model(data.x, data.edge_index, data.edge_attr)
				pred = pred[y01_indices]

				loss = crit(pred, y01.view(-1, 1))
				loss.backward()

				nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
			except Exception as e:
				if "CUDA out of memory" in str(e):
					continue
				else:
					raise e

			assert(loss != None)

			pred_class = None
			with torch.no_grad():
				pred_class = (pred >= 0.5).int().flatten() # torch.argmax(pred, dim=1)
			
			all_target += y01.cpu().numpy().tolist()
			all_pred_class += pred_class.cpu().numpy().tolist()

			total_loss += loss.item() * y01.shape[0] # data.y.shape[0]
			total_var_cnt += y01.shape[0] # data.y.shape[0]

			if cnt + hyper_params["batch_size"] <= len(dataset_train):
				pbar.update(hyper_params["batch_size"])
				cnt += hyper_params["batch_size"]
			else:
				pbar.update(len(dataset_train) - cnt)
				cnt = len(dataset_train)

	c = confusion_matrix(all_target, all_pred_class, labels=[0, 1])
	
	if hyper_params["pretrain"]:
		log_file.write("confusion_matrix_on_pretraining_set\n")
	else:
		log_file.write("confusion_matrix_on_finetuning_set\n")

	log_file.write(str(c) + "\n")

	if total_var_cnt > 0:
		log_file.write(f"train_loss={total_loss / total_var_cnt}\n")
		return total_loss / total_var_cnt
	else:
		log_file.write(f"total_var_cnt={total_var_cnt}\n")
		return None


def evaluate(log_file):
	global model
	model.eval()

	total_loss = 0
	total_var_cnt = 0

	all_target = []
	all_pred_class = []

	with torch.no_grad():
		for data in tqdm(vld_loader):
			
			pred = None
			y01_indices = None
			y01 = None
			try:
				data = data.cuda()
				model = model.cuda()
				
				y01_indices = (data.y != 2).nonzero(as_tuple=True)

				y01 = data.y[y01_indices].float()

				u0 = torch.sum((y01 == 0).int()).cpu()
				u1 = torch.sum((y01 == 1).int()).cpu()
				u01 = u0 + u1
				assert(u01 == y01.shape[0])
				weight = torch.zeros(u01).cuda()
				weight[y01 == 0] = u01 / (2 * (u0 + 1))
				weight[y01 == 1] = u01 / (2 * (u1 + 1))
				crit = BCELoss(weight=weight.view(-1, 1))

				pred = model(data.x, data.edge_index, data.edge_attr)
			except Exception as e:
				# if cuda out of memory, use CPU to do model inference
				# (or you may choose to ignore the data point causing cuda out of memory)
				if "CUDA out of memory" in str(e):
					data = data.cpu()
					model = model.cpu()
					
					y01_indices = (data.y != 2).nonzero(as_tuple=True)
					y01 = data.y[y01_indices].float()

					u0 = torch.sum((y01 == 0).int()).cpu()
					u1 = torch.sum((y01 == 1).int()).cpu()
					u01 = u0 + u1
					assert(u01 == y01.shape[0])
					weight = torch.zeros(u01).cpu()
					weight[y01 == 0] = u01 / (2 * (u0 + 1))
					weight[y01 == 1] = u01 / (2 * (u1 + 1))
					crit = BCELoss(weight=weight.view(-1, 1))
					pred = model(data.x, data.edge_index, data.edge_attr)
				else:
					raise e

			pred = pred[y01_indices] 			
			loss = crit(pred, y01.view(-1, 1))

			total_loss += y01.shape[0] * loss.item()
			total_var_cnt += y01.shape[0]

			pred_class = (pred >= 0.5).int().flatten() 

			all_target += y01.cpu().numpy().tolist()
			all_pred_class += pred_class.cpu().numpy().tolist()

	c = confusion_matrix(all_target, all_pred_class, labels=[0, 1])

	log_file.write("confusion_matrix_on_validation_set\n")
	log_file.write(str(c) + "\n")

	loss = total_loss / total_var_cnt
	precision = precision_score(all_target, all_pred_class, average='micro')
	recall = recall_score(all_target, all_pred_class, average='micro')
	f1 = f1_score(all_target, all_pred_class, average='micro')
	log_file.write("validation_loss=%.4f\n" % loss)
	log_file.write("validation_precision=%.4f\n" % precision)
	log_file.write("validation_recall=%.4f\n" % recall)
	log_file.write("validation_f1=%.4f\n" % f1)

	return loss, precision, recall, f1


best_f1 = 0
if hyper_params["checkpoint_path"] is not None and os.path.isfile(hyper_params["checkpoint_path"]):
	# evaluate the model if it is loaded from a checkpoint
	with open(hyper_params["log_dir"] + "/gnn-load.log", "w") as log_file:

		localtime = time.asctime(time.localtime(time.time()))
		log_file.write(str(localtime) + "\n")

		log_file.write("evaluate loaded model on vld set\n")

		print("evaluate loaded model on vld set first")
		_, _, _, f1 = evaluate(log_file)
	
	best_f1 = f1
	if hyper_params["pretrain"]:
			torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			}, "models/pretrain/pretrain-best.ptg")
	else:
		torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		}, "models/finetune/finetune-best.ptg")


# training loop
for epoch in range(hyper_params["epoch_num"]):
	print(f"epoch {epoch}\ntrain:")
	with open(hyper_params["log_dir"] + f"/gnn-{epoch}.log", "w") as log_file:

		localtime = time.asctime(time.localtime(time.time()))
		log_file.write(str(localtime) + "\n")

		log_file.write(f"epoch: {epoch}\n")
		train_loss = train(log_file)
		localtime = time.asctime(time.localtime(time.time()))
		
		print("eval:")
		_, _, _, f1 = evaluate(log_file)

	if hyper_params["pretrain"]:
		torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			}, "models/pretrain/pretrain-" + str(epoch) + ".ptg")
	else:
		torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			}, "models/finetune/finetune-" + str(epoch) + ".ptg")

	if f1 > best_f1:
		best_f1 = f1
		if hyper_params["pretrain"]:
			torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			}, "models/pretrain/pretrain-best.ptg")
		else:
			torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			}, "models/finetune/finetune-best.ptg")

print("Done")