REM NUMBER - FILE NUMBER WITH THE BEST WEIGHTS
SET /A NUMBER = 0
python -m FasterRCNN --eval --backbone=resnet152 --load-from=./models/weights_tune.pth_%NUMBER%
python -m FasterRCNN --backbone=resnet152 --load-from=./models/weights_tune.pth_%NUMBER% --predict-all=test