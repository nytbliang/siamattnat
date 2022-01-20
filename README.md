# SiamAttnAT

**SiamAttnAT** hosts the code for implementing the SiamAttnAT algorithm for visual tracking. The code based on the PySOT.
# Installation
Please find installation instructions in INSTALL.md.
# Quick Start: Using SiamAttnAT
**Add SiamAttnAT to your PYTHONPATH**
export PYTHONPATH=/path/to/siamattnat:$PYTHONPATH
**Download models**
Download models in Model Zoo and put the model.pth in the correct directory in experiments
**Download testing datasets**
Download datasets and put them into testing_dataset directory.
**Test tracker**
```bash
cd experiments/siamattnat_r50_l234
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
  ```
**Eval  tracker**
assume still in experiments/siamattnat_r50_l234
```bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
  ```
**Training**
See TRAIN.md for detailed instruction.

SiamAttnAT is released under the [Apache 2.0 license](https://github.com/nytbliang/siamattnat/blob/master/LICENSE). 
