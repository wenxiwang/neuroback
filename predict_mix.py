from predict import predict_mix, merge_wcc_preds


print("predict backbone on GPU (with cuda) and CPU")
pt_dir_path = "./data/pt/test/processed"
model_path = "./models/finetune/finetune-best.ptg"
res_dir_path = "./prediction/mix/wcc_predictions"
merge_dir_path = "./prediction/mix/cmb_predictions"

predict_mix(pt_dir_path, model_path, res_dir_path)
merge_wcc_preds(res_dir_path, merge_dir_path)
