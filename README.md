# HAR_DFT
A Lightweight Deep Learning Solution for mmWave Human Activity Recognition based on Discrete Fourier Transformation

## Data Preoricessing
```
python voxel_generator.py --frame 60 --sliding_window 10 --data_path data/raw/ --data_save data/voxel/
```

## Model Train
```
python train.py --FeatureExt FC --DFT 1 --DFT_number 16  --frame 60 --data_path data/voxel/ --model_dir model_data/ --learning_rate 0.0001 --beta_1 0.9 --beta_2 0.999 --checkpoint_monitor val_accuracy --checkpoint_mode max --batch_size 15 --epochs 15 --draw 1
```

## Model Test
```
python train.py --FeatureExt FC --DFT 1 --DFT_number 16  --frame 60 --data_path data/voxel/ --model_dir model_data/ --confusion_metrics 1 --FLOPS 1
```
