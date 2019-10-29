# Severstal
Scripts for Severstal Kaggle Competition

## Training phase
1. Train a model in a script under FixCheck branch

    cd Severstal/src
    
    python main.py --model 'se_resnet50' --sch 2 --loss 2 --output 2 --augment 2 -e 40 --wlovasz 0.2
    
  
  You will expect the first few epochs result similiar to
  
    Epoch 0 :Train_loss:1.373 Train_dice:0.465 Train_other:0.849 Valid_loss:0.855 Valid_dice:0.782 Valid_other:0.854
    Improving val_dice from 0.782 to 0.867, saving the model

    Epoch 1 :Train_loss:0.661 Train_dice:0.852 Train_other:0.879 Valid_loss:0.536 Valid_dice:0.867 Valid_other:0.913
    Improving val_dice from 0.867 to 0.892, saving the model

    Epoch 2 :Train_loss:0.474 Train_dice:0.874 Train_other:0.901 Valid_loss:0.435 Valid_dice:0.892 Valid_other:0.913
    Improving val_dice from 0.892 to 0.906, saving the model

## Inference Phase
1. Execute the code
    cd Severstal/src
    
    python main.py --load_mod --model 'se_resnet50' --sch 2 --loss 2 --output 2 --augment 2 -e 40 --wlovasz 0.2
  

2. The second command will evaluate the model with postprocessing methods
    
        Category 1: Mean 0.9749, True Area[Neg,0.0000; Pos,0.0127], Pred Dice[Neg,0.9979; Pos,0.6183], Dice Diff[Neg,4.000; Pos,46.562]
        
        Category 2: Mean 0.9796, True Area[Neg,0.0000; Pos,0.0093], Pred Dice[Neg,1.0000; Pos,0.0000], Dice Diff[Neg,0.000; Pos,41.000]
        
        Category 3: Mean 0.8690, True Area[Neg,0.0000; Pos,0.0634], Pred Dice[Neg,0.9750; Pos,0.7240], Dice Diff[Neg,29.000; Pos,234.348]
        
        Category 4: Mean 0.9817, True Area[Neg,0.0000; Pos,0.0870], Pred Dice[Neg,0.9989; Pos,0.7346], Dice Diff[Neg,2.000; Pos,34.773]


        Final SWA Dice 0.951
        
        ==============SWA Predict===============
        
        Four labels ratio (1~4): 0.057,0.000,0.416,0.063
        Label Num (Zero labels~four Labels): 999,946,66,0,0
        Label ratio (Zero labels~four Labels): 0.4968,0.4704,0.0328,0.0000,0.0000
        
        ==============True===============
        
        Four labels ratio (1~4): 0.061,0.020,0.422,0.065
        Label Num (Zero labels~four Labels): 941,997,73,0,0
