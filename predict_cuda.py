from predict import predict_cuda, merge_wcc_preds


print("predict backbone on GPU (with cuda)")
pt_dir_path = "./data/pt/test/processed"
model_path = "./models/finetune/finetune-best.ptg"
res_dir_path = "./prediction/cuda/wcc_predictions"
merge_dir_path = "./prediction/cuda/cmb_predictions"

predict_cuda(pt_dir_path, model_path, res_dir_path)
merge_wcc_preds(res_dir_path, merge_dir_path)
