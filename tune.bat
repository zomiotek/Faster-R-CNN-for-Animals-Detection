REM NUMBER - FILE NUMBER WITH THE BEST WEIGHTS
SET /A NUMBER = 0
python -m FasterRCNN --train --no-augment --backbone=resnet152 --log-csv=./log/training_metrics_2.csv --learning-rate=1e-4 --epochs=15 --load-from=./models/weights_train.pth_%NUMBER% --save-best-to=./models/weights_tune.pth
