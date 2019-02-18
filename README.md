# Recursive Networks

**Note**:  
Previous work on analysis of saturation of layers using recursivity has moved [this other repository][saturation].  
Also, there is a similar analysis covering weight initizalization sensitivity [here][initialization]

---

Empirical analysis on ["Understanding Deep Architectures using a Recursive Convolutional Network"][paper]   

![recursive][recursive_img]

Get insights of how recursive networks works, under which circusntances outperform original models and how they could serve as a simulation on a increase of the network budget (number of parameters).   

## Original Networks

#### Fully Connected Network
- TBI  

#### Convolutional Network
![][untied_model]

- 16 conv layers with 32 filters
- ReLU activation function

[saturation]: https://github.com/PabloRR100/Distilling-Deep-Networks.git
[initialization]: https://github.com/PabloRR100/NN_Initialization_Sensitivity.git

[recursive_img]: images/recursive.png
[recursiveanalysis]: images/recursive_h2_w4.png
[untied]: images/untied_model.png
[paper]: https://arxiv.org/abs/1312.1847
