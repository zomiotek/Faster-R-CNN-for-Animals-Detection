python -m FasterRCNN --train --no-augment --backbone=resnet152 --log-csv=./log/training_metrics_1.csv --learning-rate=1e-3 --epochs=15 --save-best-to=./models/weights_train.pth
