# DI-GAN
* This is the source code of Pan-sharpening Approach via Two-stream Details Injection Based on Generative Adversarial Networks (DI-GAN) using Keras:


# license
* Copyright (c) 2019 Digital Research Centre of Sfax (CRNS).
* All rights reserved. This source code should be used for nonprofit purposes only.

# Usage
* Before running the code, you have to create a MAT-FILE, that includes the MS image (I_MS), the associated PAN image (I_PAN), and sensor variable (WV3, WV2, QB, GeoEye-1, or IKONOS).
* In order to train the network:

```
python DI-GAN-TRAIN.py train_image_path
```
* To produce the pansharpening product at reduced or full-resolution:
```
python DI-GAN-FULL.py mode input_path
```
* where mode is full for high resolution, or reduced for reduced scale
* input_path: the input image directory.





