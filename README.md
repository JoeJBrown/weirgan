# WEIRGAN

Python 3.5
Tensorflow 1.2

Save the dataset in a directory named ./data and the --dataset argument is the name of the file the data is saved in.

## Train
```
python main.py --dataset=cifar --use_gpu=True --input_scale_size=32 --cat_num=2 --cont_num=5
```

To load a model, the file must be located in the ./logs folder.

## Test
```
python main.py --dataset=cifar --load_path=path_to_model --use_gpu=True --is_train=False --input_scale_size=32 --cat_num=2 --cont_num=5
```

Further hyper-parameters can be tuned, check the config file for more detail.


Credit to carpedm20 for their DCGAN implementation which helped provide the base code for this project. Their implementation can be found [Here](https://github.com/carpedm20/DCGAN-tensorflow).
