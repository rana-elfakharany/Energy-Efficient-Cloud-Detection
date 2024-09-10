# Energy-Efficient Cloud Detection on Satellites Using Edge-Based Deep Learning

## Abstract
Cloud detection in remote sensing imagery has been a prominent field of study to enhance the quality of satellite data and optimize the image transmission process from the satellite to the base station. The presence of cloud cover in satellite images hinders the actual image being captured and results in degraded images. Transmitting such images to the ground station can lead to an unnecessary consumption of expensive scarce resources like power and bandwidth. Despite the improvements in the existing cloud detection methods, the available models are large, meaning they cannot be easily deployed on an edge device like a satellite. This paper focuses on implementing an edge-enabled deep learning cloud detection method. Using Landsat 8 images for training, the proposed U-Net architecture achieved an accuracy of 0.934, an F1–score of 0.881, and an IoU of 0.793. The proposed architecture achieves comparable accuracy to state-of-the-art methods with a much smaller size. The system was deployed on edge devices including Raspberry Pi 5, Jetson Nano, and Coral Dev. Jetson Nano performed the best with a power consumption of 6.153 W and inference time of 0.0062 seconds/image. 

## Notes
In this repository, you will find the code for the four different models discussed within the paper: U-Net, U-Net++, Inception Net, and Deeplabv3. 
1. Due to the limited memory capacity, we had to manually run k-fold instead of doing it all in one run. Therefore, the code is structured as so. However, it is possible to edit the code according to your preference.
2. To run the code on the selected hardware devices, the models had to be quantized to TensorFlow Lite or TensorRT, depending on the device.
