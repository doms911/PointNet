download-modelnet:
	python scripts/download_modelnet10.py

download-shapenet:
	python scripts/download_shapenetpart.py

train-cls:
	python -m train_pointnet_cls \
		--data_root data/raw/ModelNet10 \
		--num_classes 10 \
		--bn \
		--cache_dir ./cache_modelnet10

predict-shapenet:
	python predict.py --ckpt results/pointnet_modelnet10.pt --input data/raw/ModelNet10/bathtub/test/bathtub_0107.off

confusionmatrix-shapenet:
	python -m src.utils.confusion_matrix \
		--ckpt results/pointnet_modelnet10.pt \
		--data_root data/raw/ModelNet10 \
		--split test \
		--batch_size 16 \
		--cache_dir ./cache_modelnet10

train-seg:
	python train_pointnet_seg.py \
		--data_root data/raw/shapenetcore_partanno_segmentation_benchmark_v0 \
		--out_dir results \
		--cache_dir cache_shapenetpart \
		--batch_size 16 \
		--npoints 1024 \
		--workers 6

predict-seg:
	python predict_seg.py \
		--ckpt results/pointnet_shapenetpart_best.pt \
		--data_root data/raw/shapenetcore_partanno_segmentation_benchmark_v0 \
		--split val --index 0

eval-seg:
	python eval_shapenetpart.py \
		--ckpt results/pointnet_shapenetpart_best.pt \
		--data_root data/raw/shapenetcore_partanno_segmentation_benchmark_v0 \
		--cache_dir cache_shapenetpart \
		--split val


