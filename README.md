# Severstal
Scripts for Severstal Kaggle Competition

1. Run the script under FixCheck branch

    cd Severstal/src
    python main.py --model 'se_resnet50' --sch 2 --loss 2 --output 2 --augment 2 -e 40 --wlovasz 0.2
    python main.py --load_mod --model 'se_resnet50' --sch 2 --loss 2 --output 2 --augment 2 -e 40 --wlovasz 0.2
  
  You will expect the first epoch result similiar to
  
    Epoch 0 :Train_loss:1.373 Train_dice:0.465 Train_other:0.849 Valid_loss:0.855 Valid_dice:0.782 Valid_other:0.854
    Improving val_dice from 0.782 to 0.867, saving the model

    Epoch 1 :Train_loss:0.661 Train_dice:0.852 Train_other:0.879 Valid_loss:0.536 Valid_dice:0.867 Valid_other:0.913
    Improving val_dice from 0.867 to 0.892, saving the model

    Epoch 2 :Train_loss:0.474 Train_dice:0.874 Train_other:0.901 Valid_loss:0.435 Valid_dice:0.892 Valid_other:0.913
    Improving val_dice from 0.892 to 0.906, saving the model
