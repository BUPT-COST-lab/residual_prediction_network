# residual_prediction_network
This is a TensorFlow implementation of the paper as follows: 

Liu X, Yin J. Stacked residual blocks based encoder–decoder framework for human motion prediction[J]. Cognitive Computation and Systems, 2020, 2(4): 242-246.

## Setup
Required python libraries: tensorflow (>=1.0) + opencv + numpy.
Tested in ubuntu/centOS + nvidia titan X (Pascal) with cuda (>=8.0) and cudnn (>=5.0).


## Training
Use the `scripts/ResNet_short_term_train_2.sh` to train/test by the following commands:
```shell
cd scripts
sh ResNet_short_term_train_2.sh  
```

## Citation
If you use this code for your research, please consider citing:
```latex
@article{liu2020stacked,
  title={Stacked residual blocks based encoder--decoder framework for human motion prediction<? show [AQ="" ID=" Q1]"?},
  author={Liu, Xiaoli and Yin, Jianqin},
  journal={Cognitive Computation and Systems},
  volume={2},
  number={4},
  pages={242--246},
  year={2020},
  publisher={IET}
}
```

## Contact
A part of code adopt from PredCNN at https://github.com/xzr12/PredCNN.git. 

