# Recursive Networks

**Note**:  
Previous work on analysis of saturation of layers using recursivity has moved [this other repository][saturation].  
Also, there is a similar analysis covering weight initizalization sensitivity [here][initialization]

---

Empirical analysis on ["Understanding Deep Architectures using a Recursive Convolutional Network"][paper]   

![paper_img][paper_img]

Get insights of how recursive networks works, under which circusntances outperform original models and how they could serve as a simulation on a increase of the network budget (number of parameters).   

## Original Network

- Dataset: CIFAR10
- Batch Size: 128
- Optimizer = SGD
- Learning Rate: 0.001

![single][single_img]

Number of Parameters = (8 · 8 · 3 · M) + (3 · 3 · M^2 · L) + (M · (L + 1)) + (64 · M · 10 + 10)
- **CASE:** L = 16, M = 32 --> Parameters = 174,634

### Recursive Implementation:
For the recursive network, the term L for the number of layers dissapear, since just 1 is contributing to the total number of parameteres, and its being reused to add non_linearities and increase the depth of the network.

![recursive][recursive_img]

Number of Parameters = (8 · 8 · 3 · M) + (3 · 3 · M^2) + (2 · M) + (64 · M · 10 + 10)
- **CASE:** L = 16, M = 32 --> Parameters = 35.914  --> ***Ensemble size allowed = 5***

### Custom Recursive Implementation:
One bottleneck we find in the recursive implementation, is that 10% of the total number of the parameters lay on the classification matrix.  
Theefore, we propose an alternative to include one more convolutional layer before the flattening with less output channels *(F, where F<M)*.  
This extra layers is a new source of parameters, but reduces the size of the classification matrix, since now depends on *F*. The only constraint to ensure a reduction in the total number of parameters is that: *640M > 9F^2 + 642F*.

![recursive_custom][custom_recursive_img]

Number of Parameters = (8 · 8 · 3 · M) + (3 · 3 · M^2) + (2 · M) + (64 · M · 10 + 10)
- **CASE:** L = 16, M = 32 --> Parameters = 28.010 --> ***Ensemble size allowed = 6***

***Not only we have reduce the proportion of parameters in the fully connected layer, but we can use a bigger ensemble size.***


[saturation]: https://github.com/PabloRR100/Distilling-Deep-Networks.git
[initialization]: https://github.com/PabloRR100/NN_Initialization_Sensitivity.git

[paper_img]: ./images/recursive.png
[simple_img]: ./images/01_single_model.png
[recursive_img]: ./images/02_recursive_model.png
[custom_recursive_img]: ./images/03_custom_recursive_model.png


[recursiveanalysis]: ./images/recursive_h2_w4.png

[paper]: https://arxiv.org/abs/1312.1847
