This repo contains file to Fine-Tune resnet34 model for [GTSRB task](https://pytorch.org/vision/main/generated/torchvision.datasets.GTSRB.html).

[ResNetTransf.ipynb](https://github.com/StarLord202/GTSRB/blob/master/ResNetTransf.ipynb) contains train process and it's results

and you also can test the model via [inference.py](https://github.com/StarLord202/GTSRB/blob/master/inference.py) file, but firstly download the weights via [this link](https://drive.google.com/file/d/1Gs8mJBBfdBofuBQdViNrPAARUCoBWzNM/view?usp=drive_link)

and add it to the directory where [inference.py](https://github.com/StarLord202/GTSRB/blob/master/inference.py) is placed

Test folder contains images to test model perfomance, those images are not from GTSRB dataset.

In order to test model firstly download all requirments by using following command  ```pip install -r requirements.txt```
Then use following commands to run tests ```python inference.py --image_dir <path to your dir>``` 
