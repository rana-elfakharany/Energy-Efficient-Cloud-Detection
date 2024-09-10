# Energy-Efficient Cloud Detection on Satellites Using Edge-Based Deep Learning

## Introduction
Cloud detection in remote sensing imagery has been a prominent field of study to enhance the quality of satellite data and optimize the image transmission process from the satellite to the base station. The presence of cloud cover in satellite images hinders the actual image being captured and results in degraded images. Transmitting such images to the ground station can lead to an unnecessary consumption of expensive scarce resources like power and bandwidth. Despite the improvements in the existing cloud detection methods, the available models are large, meaning they cannot be easily deployed on an edge device like a satellite. 

After testing different models, the proposed U-Net architecture obtained an accuracy of 0.934, an F1-score of 0.881, and an IoU of 0.793 using training photos from Landsat 8. The suggested design is significantly smaller while achieving precision on par with cutting-edge techniques. The Raspberry Pi 5, Jetson Nano, and Coral Dev were among the edge devices on which the system was installed. With an inference time of 0.0062 seconds per image and a power consumption of 6.153 W, the Jetson Nano demonstrated the best performance. 


## Notes
In this repository, you will find the code for the four different models discussed within the paper: U-Net, U-Net++, Inception Net, and Deeplabv3. 
1. Due to the limited memory capacity, we had to manually run k-fold instead of doing it all in one run. Therefore, the code is structured as so. However, it is possible to edit the code according to your preference.
2. To run the code on the selected hardware devices, the models had to be quantized to TensorFlow Lite or TensorRT, depending on the device.
