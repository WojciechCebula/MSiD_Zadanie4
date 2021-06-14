# MSiD_4

## Introduction
The project contains methods and research concerning the fashion picture recognition task. All the files that are needed to train and test models are included in the https://github.com/zalandoresearch/fashion-mnist repository. This project is not an application. It was design to validate different models and score their accuracy.

## Methods
k-NN method and neural networks are used in this project. We can basically divide this paragraph into three separate pieces, one related to Gabor filters, one to k-NN and one to neural networks.

### Gabor filters
Gabor filters are widely used for image pre-processing, to specify features that are in the picture. There are different parameters that can be used to create a filter, like kernel size etc. and in here the maximal number of all possible combinations of parameters that we will be using is 384. This number represents how many different filters will be generated.

Here are some exaples of kernels I fould in the kernel-set.
</br><img src="https://user-images.githubusercontent.com/67602274/121958394-ed340d80-cd63-11eb-93a3-af85d688710b.png" height=300/>
<img src="https://user-images.githubusercontent.com/67602274/121959109-a0046b80-cd64-11eb-9f79-06931a313e3b.png" height=300/>
<img src="https://user-images.githubusercontent.com/67602274/121959621-510b0600-cd65-11eb-8105-00bc0a3bf744.png" height=300/>

As you can see different kernels focus on looking for different features, for example the one on the left is searching for horizontal lines.

### k-NN
For k-NN algorithm we will be using Gabor filters and test which one of them is best suited for fasion dataset. We are also testing the best k value for our model, ranging from 1 to 1000 (not all numbers in between are included).

### Neural networks
There are basically two types of neural networks tested in this project. Simple deep neural network and a convolutional one. This all comes along with some methods for data augmentation like salt and pepper, or dropout layers.

## Results
### k-NN
For k-NN algorithm almost all top 20 Gabor filters are focusing on vertical lines, and they are all 3 by 3 filters. Here are a few best models, along with all the parameters used for that model.
| Accuracy  | Filter number | Kernel size | theta | sigma | lambda | gamma |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.8588 | 195  | 3x3 | 3.141592653589793 | 1 | 0.7853981633974483 | 0.5 |
| 0.8569 | 217  | 3x3 | 3.141592653589793 | 5 | 0.7853981633974483 | 0.05 |
| 0.8568 | 205  | 3x3 | 3.141592653589793 | 3 | 0.7853981633974483 | 0.05 |

Here we have the accuracy for different k parameters, for the best filter (id: 195)
| K-value  | 1 | 3 | 5 | 10 | 15 | 20 | 30 | 50 | 100 | 200 | 500 | 1000 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Acurracy  | 0.855 | 0.8581 | ***0.8588*** | 0.8539 | 0.8497 | 0.8488 | 0.8456 | 0.8384 | 0.8253 | 0.8129 | 0.789 | 0.7695 |

As you can see for k=5 our accuracy is the highest.

The best filter:
</br><img src="https://user-images.githubusercontent.com/67602274/121962192-b44a6780-cd68-11eb-9a10-3bbccad02592.png" height=300/>

For k-NN algorithm the best solution I found on benchmark site was one with an accuracy of 0.860 which is slightly higher than what we achieved. Sadly, our model needs around 1.5 hour to train itself, which is almost double the time of the algorithm I found online.

### Neural networks
Here I also tried using filters for our dataset, but this time even for the simple DNN model the accuracy was lower than without using any filters. Besides filters, I used salt and pepper data augmentation algorithm, and I created many types of neural networks with different number of neurons in each layer [64, 128, 256, 512, 1024] and with different number of hidden layers [1, 3, 5, 10]. We are using Adams optimizer, sparse categorical crossentropy loss function, early stopping along with reducing the length of our learning step. Here are some of the best results.
| Accuracy  | Filter | Number of hidden layers | Number of Neurons | Salt&Pepper |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.8848999738693237 | No  | 3 | 1024 | Yes |
| 0.8834999799728394 | Yes  | 5 | 1024 | No |
| 0.8826000094413757 | No  | 10 | 1024 | Yes |

For the convolutional network, there is only one I prepared. It is using RMSprop and Adam optimizer, sparse categorical crossentropy loss function, Salt&Pepper algorithm, batch normalization, dropout layers, early stopping, reducing the length of its learning step.    
Model:
</br><img src="https://user-images.githubusercontent.com/67602274/121968776-393a7e80-cd73-11eb-8056-74d33730d17b.png" height=500/> 

Accuracy for different parameters:
| Accuracy  | Optimizer | Salt&Pepper |
| ------------- | ------------- | ------------- |
| 0.9342 | Adam  | Yes |
| 0.9329 | RMSprop  | Yes |
| 0.9321 | RMSprop  | No |
| 0.9301 | Adam  | No |

If we compare that to some models that are posted on Fashion repository (like @khanguyen1207 model Conv+pooling+BN -> 0.934 accuracy), we can see that its accuracy is almost the same as in our model.

## Usage
All the functions that I was using for this project we can run straight from main.py file, by calling them in __main__. The fashion data should be stored in /data/fashion/ folder. Filtered testing data goes to /data/filters/test/ folder and training data to /data/filters/train/ folder. Libraries used for this project that are needed to be included in the environment are: tensorflow, numpy, multiprocessing, cv2, sklearn, gzip and os.

## References
https://en.wikipedia.org/wiki/Gabor_filter
https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
https://keras.io/api/callbacks/reduce_lr_on_plateau/
https://www.tensorflow.org/tutorials/images

 
