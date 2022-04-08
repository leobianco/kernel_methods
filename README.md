# Image classification with kernel methods
Implementation done concerning the data challenge of the course Kernel Methods 
at the master MVA. The objective was to classify images from a subset of the 
CIFAR-10 dataset using exclusively kernel methods and our own code, i.e., 
without the aid of libraries such as libsvm or sklearn. 

To test our methods simply run ```start.py```. If you press enter at all 
prompts you will recover an "one vs. all" SVM with C=10, RBF kernel of unit
variance, trained on half the data. This gives us roughly 50% accuracy.
