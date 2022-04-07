#python3 -u test_erfnet.py CULane ERFNet train test_img \
#                          --lr 0.01 \
#                          --gpus 4 5 6 7 \
#                          --npb \
#                          --resume trained/ERFNet_trained.tar \
#                          --img_height 208 \
#                          --img_width 976 \
#                          -j 10 \
#                          -b 5

python -m test_erfnet CULane ERFNet train test_img \
                          --lr 0.01 \
                          --gpus 4 5 6 7 \
                          --npb \
                          --resume trained/ERFNet_trained.tar \
                          --img_height 208 \
                          --img_width 976 \
                          -j 10 \
                          -b 5


#Namespace(arch='resnet101', batch_size=5, dataset='CULane', dropout=0.1, epochs=24, eval_freq=1, evaluate=False, gpus=[4, 5, 6, 7], img_height=208, img_width=976, local_rank=0, lr=0.01, lr_steps=[10, 20], method='ERFNet', momentum=0.9, no_partialbn=True, print_freq=1, resume='trained/ERFNet_trained.tar', snapshot_pref='', start_epoch=0, test_size=840, train_list='train', train_size=840, val_list='test_img', weight='', weight_decay=0.0001, workers=10)
