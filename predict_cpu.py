from predict import predict_cpu, merge_wcc_preds


print("predict backbone on CPU")
pt_dir_path = "./data/pt/test/processed"
model_path = "./models/finetune/finetune-best.ptg"
res_dir_path = "./prediction/cpu/wcc_predictions"
merge_dir_path = "./prediction/cpu/cmb_predictions"

predict_cpu(pt_dir_path, model_path, res_dir_path)
merge_wcc_preds(res_dir_path, merge_dir_path)