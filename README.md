# E2E-ULDSC
End-to-End Ultra-Lightweight DSC (E2E-ULDSC) neural network

Introduction and Requirement

In our project, the Environmental Sound Classification on ESC-50 dataset is realized by the proposed E2E-ULDSC neural network, which mainly consists of 8 depthwise separable convolution layers for lightweight purpose. The number of parameters and FLOPs are reduced to 0.156M and 0.125G, respectively, while keeping the accuracy of 87.25%.
The E2E-ULDSC-Sparse is obtained through knowledge distillation and model pruning. The number of parameters and FLOPs are further reduced to 0.01M and 0.009G, while the accuracy reaches 84.50%.


1. Requirements:
   
   python==3.8.17
   
   pytorch==2.3.0
   
   numpy==1.23.5
2. Dataset preparation
   ESC-50 contains 2000 samples (5-sec-long audio recordings, sampled at 16kHz and 44.1kHz) which are equally distributed over 50 balanced disjoint classes (40 audio samples for each class).On this basis, we changed the sampling rate of the data set to 20kHZ, and applied five-fold cross-validation processing. Each fold contains 4000 pieces of data (1,1,30225), four of which are used for network training, and the other one is used for validation.Download and process the data set by using the following project:https://github.com/mohaimenz/acdnet   
All the required data of ESC-50 for processing 20kHz are now ready at datasets/esc50 directory
3. How to test
   3.1. Download or clone this repositoriesLoading
   
   3.2 run "tester.py";
   
   3.3. Enter the model path: "$Path\E2E_ULDSC.pt"
   
   3.4. Select the fold on which the model was Validated: "4"
   
   Results: No. Param: 0.156M; FLOPs: 0.125G; Top-1 Accuracy: 87.25%
  
5. E2E_ULDSC.pt: the original model without channel pruned and without weight pruned.

   E2E_ULDSC_Sparse.pt: the model with 20% channel pruned and with 90% weight pruned:

   Results: No. Param: 0.01M; FLOPs: 0.009G; Top-1 Accuracy: 84.50%  

    
  





