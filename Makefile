run_docker:
	docker run -it --rm --gpus all -v `pwd`:/work/ shintomi_blockout_ai bash
train_test:
	python3 ./ai/train.py --train_dataset ./data/MNIST-resized/train --test_dataset ./data/MNIST-resized/test --train_label ./data/MNIST-resized/train_labels.csv --test_label ./data/MNIST-resized/test_labels.csv --save-model | tee ./log/train.log

