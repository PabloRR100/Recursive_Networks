# Recursive Networks

**Note**:  
Previous work on analysis of saturation of layers using recursivity has moved [this other repository][saturation].  
Also, there is a similar analysis covering weight initizalization sensitivity [here][initialization]

---

Empirical analysis on ["Understanding Deep Architectures using a Recursive Convolutional Network"][paper]   

![recursive][recursive_img]

Get insights of how recursive networks works, under which circusntances outperform original models and how they could serve as a simulation on a increase of the network budget (number of parameters).   

## Original Network

- Dataset: CIFAR10
- Batch Size: 128
- Optimizer = SGD
- Learning Rate: 0.001

The single model consists on L layers of "Same Pading" convolutions with M filters and a classification matrix C, consisting on a fully connected layer from the flattened vector to the 10 classes.  

The network is shown here:

![untied][untied_model]

### Recursive Convolutional Network

The recursive model 



Number of Parameters = (8 · 8 · 3 · M) + (3 · 3 · M^2 · L) + (M · (L + 1)) + (64 · M · 10 + 10)

- **CASE I:** 16 conv layers with 32 filters --> Parameters = 174,634

[saturation]: https://github.com/PabloRR100/Distilling-Deep-Networks.git
[initialization]: https://github.com/PabloRR100/NN_Initialization_Sensitivity.git

[recursive_img]: ./images/recursive.png
[recursiveanalysis]: ./images/recursive_h2_w4.png
[untied_model]: images/untied_model.png
[paper]: https://arxiv.org/abs/1312.1847
