# E2E-ULDSC
end-to-end ultra-lightweight DSC neural network

In our project, the environmental sound classification for ESC-50 data set is realized. Through knowledge distillation and model pruning, the parameters and FLOP of the model are greatly reduced, and the accuracy rate reaches 84%.

ESC-50 contains 2000 samples (5-sec-long audio recordings, sampled at 16kHz and 44.1kHz) which are equally distributed over 50 balanced disjoint classes (40 audio samples for each class).On this basis, we changed the sampling rate of the data set to 20kHZ, and applied five-fold cross-validation processing. Each fold contains 4000 pieces of data (1,1,30225), four of which are used for network training, and the other one is used for validation.


