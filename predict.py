import torch
from torch.cuda.amp import autocast

from data import *
from tqdm import *
from gt_model import GTModel

import os
import gc
import time
import tarfile


def predict_single(pt_dir_path, pt_file, model_path, res_dir_path, is_cuda=True):
	data = torch.load(os.path.join(pt_dir_path, pt_file))

	reverse = data.edge_index.index_select(0, torch.LongTensor([1, 0]))
	data.edge_index = torch.cat([data.edge_index, reverse], dim=1)
	data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
	data.x = data.x.float()
	data.edge_index = data.edge_index.long()
	data.edge_attr = data.edge_attr.float()

	mymodel = GTModel(3,3)
	name_encoder = None
	name_emb = None
	if is_cuda:
		data = data.cuda()

		checkpoint = torch.load(model_path, map_location=torch.device('cuda'))

		mymodel.load_state_dict(checkpoint['model_state_dict'])
		mymodel = mymodel.cuda()
		clean_fn = ".".join(pt_file.split(".")[:-3])
			 
	else:
		data = data.cpu()

		checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

		mymodel.load_state_dict(checkpoint['model_state_dict'])

		mymodel = mymodel.cpu()

	mymodel.eval()

	with torch.no_grad():
		pred = mymodel(data.x, data.edge_index, data.edge_attr)

		n2v = data.n2v.cpu().numpy().tolist()

		cnf_file_name = pt_file[:-3]
		with open(res_dir_path + "/" + cnf_file_name + ".res", "w") as f:
			for n, v in enumerate(n2v):
				pred_score = pred[n].tolist()[0]
				f.write(f"{v},{pred_score}\n")
		
		tmp = os.getcwd()
		os.chdir(res_dir_path)
		tar = tarfile.open(cnf_file_name + ".res.tar.gz", "w:gz")
		tar.add(cnf_file_name + ".res")
		tar.close()
		os.remove(cnf_file_name + ".res")
		
		os.chdir(tmp)


def predict_mix(pt_dir_path, model_path, res_dir_path):
	if not os.path.isdir(res_dir_path):
		os.makedirs(res_dir_path)

	pt_file_lst = list(os.listdir(pt_dir_path))
	
	pt_file_lst = sorted(pt_file_lst, key=lambda pt_file: os.path.getsize(f"{pt_dir_path}/{pt_file}"))
	
	recog_cnf_files = set()
	for pt_file in pt_file_lst:
		recog_cnf_files.add(".".join(pt_file.split(".")[:-2]))

	print("# cases:", len(recog_cnf_files))

	mode = "cuda"

	if not os.path.isdir("./log/predict_mix"):
		os.makedirs("./log/predict_mix")

	with tqdm(total=len(pt_file_lst)) as pbar:
		for pt_file in pt_file_lst:
			with open(f"./log/predict_mix/{pt_file}.csv", "w") as perf_file:
				start = time.time()
				if mode == "cuda":
					try:
						predict_single(pt_dir_path, pt_file, model_path, res_dir_path, is_cuda=True)
					except Exception as e:
						print("Switch to CPU")
						mode = "cpu"

						start = time.time()
						predict_single(pt_dir_path, pt_file, model_path, res_dir_path, is_cuda=False)
				else:
					assert(mode == "cpu")
					predict_single(pt_dir_path, pt_file, model_path, res_dir_path, is_cuda=False)

				time_cost = time.time() - start # in seconds

				perf = pt_file + "," + mode + "," + str(time_cost) + "\n"
				perf_file.write(perf)

				pbar.update()

		print("Done")


def predict_cpu(pt_dir_path, model_path, res_dir_path):
	if not os.path.isdir(res_dir_path):
		os.makedirs(res_dir_path)

	pt_file_lst = list(os.listdir(pt_dir_path))
	pt_file_lst = sorted(pt_file_lst, key=lambda pt_file: os.path.getsize(f"{pt_dir_path}/{pt_file}"))

	if not os.path.isdir("./log/predict_cpu"):
		os.makedirs("./log/predict_cpu")

	with tqdm(total=len(pt_file_lst)) as pbar:
		for pt_file in pt_file_lst:
			with open(f"./log/predict_cpu/{pt_file}.csv", "w") as perf_file:
				start = time.time()
				try:
					predict_single(pt_dir_path, pt_file, model_path, res_dir_path, is_cuda=False)
				except Exception as e:
					print(pt_file, e)
					break
					
				time_cost = time.time() - start # in seconds

				perf = pt_file + ",cpu," + str(time_cost) + "\n"
				perf_file.write(perf)

				pbar.update()

		print("Done")


def predict_cuda(pt_dir_path, model_path, res_dir_path):
	if not os.path.isdir(res_dir_path):
		os.makedirs(res_dir_path)

	pt_file_lst = list(os.listdir(pt_dir_path))
	pt_file_lst = sorted(pt_file_lst, key=lambda pt_file: os.path.getsize(f"{pt_dir_path}/{pt_file}"))

	if not os.path.isdir("./log/predict_cuda"):
		os.makedirs("./log/predict_cuda")

	with tqdm(total=len(pt_file_lst)) as pbar:
		for pt_file in pt_file_lst:
			with open(f"./log/predict_cuda/{pt_file}.csv", "w") as perf_file:
				start = time.time()
				try:
					predict_single(pt_dir_path, pt_file, model_path, res_dir_path, is_cuda=True)
				except Exception as e:
					print(pt_file, e)
					break
					
				time_cost = time.time() - start # in seconds

				perf = pt_file + ",cuda," + str(time_cost) + "\n"
				perf_file.write(perf)

				pbar.update()

		print("Done")


def merge_wcc_preds(res_dir_path, merge_dir_path, rm_wcc_pred=False):
	print("merge")
	if not os.path.isdir(merge_dir_path):
		os.makedirs(merge_dir_path)

	cnf2pred = {}
	for pred_fn in os.listdir(res_dir_path):
		cnf_file_name = ".".join(pred_fn.split(".")[:-4])

		if cnf_file_name not in cnf2pred:
			cnf2pred[cnf_file_name] = []

		cnf2pred[cnf_file_name].append(pred_fn)

	for cnf_file_name, pred_fn_lst in cnf2pred.items():
		backbone_lst = []
		for pred_fn in pred_fn_lst:
			tar_file_obj = tarfile.open(f"{res_dir_path}/{pred_fn}", "r")

			# extract a file
			res_name = ".".join(pred_fn.split(".")[:-2])
			res_file = tar_file_obj.extractfile(res_name)
			
			for line in res_file.read().decode('utf-8').split("\n"):
				line = line.strip()
				if len(line) > 0:
					backbone_lst.append(line)

			#close file
			tar_file_obj.close()

		backbone_lst = sorted(backbone_lst, key= lambda l: float(l.split(",")[1]), reverse=True)
		with open(merge_dir_path + "/" + cnf_file_name + ".res", "w") as f:
			for line in backbone_lst:
				f.write(f"{line}\n")

		tmp = os.getcwd()
		os.chdir(merge_dir_path)
		tar = tarfile.open(cnf_file_name + ".res.tar.gz", "w:gz")
		tar.add(cnf_file_name + ".res")
		tar.close()
		os.remove(cnf_file_name + ".res")
		
		os.chdir(tmp)

		if rm_wcc_pred:
			for pred_fn in pred_fn_lst:
				os.remove(f"{res_dir_path}/{pred_fn}")
	print("done")