python -m test_video_erfnet CULane ERFNet train test_img \
                          --lr 0.01 \
                          --gpus 4 5 6 7 \
                          --npb \
                          --resume trained/ERFNet_trained.tar \
                          --img_height 208 \
                          --img_width 976 \
			  --video_path test4.mp4 \
                          -j 10 \
                          -b 5
