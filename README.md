# ELM-Auto-Encoder



To run multi-layer ELM
 
You have to download MNIST dataset from http://yann.lecun.com/exdb/mnist/ Follow the instructions in website to extract MNIST. and place it in a folder called "mnist"

Run the code using the following command

[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = MELM_MNIST25(0, 0, 3, [700,15000], [1e-1,1e4,1e8],0.05, [0.7,1], [0.8,0.9])

Or

[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = MELM_MNIST25(0, 0, 3, [700,15000], [1e-1,1e3,1e8],0.05, [0.7,6], [0.8,4])
