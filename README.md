# mva_DL

To run the code use the `main.py` file. The following option are available:

* --model : model to choose to perform training (resnet, vgg16, inceptionv3, mixte)
* --batch-size : input batch size for training (default: 64)
* --epochs : number of epochs to train (default: 10)
* --lr : learning rate (default: 0.01)
* --seed : random seed (default: 1)
* --log-interval : how many batches to wait before logging training status
* --experiment : folder where experiment outputs are located.

The use of Grad CAM doesn't work automatically with the selected network and requires to modified the layer used in the file `grad_2.py`