# mli-defconv

This crate implements deformable convolution. This idea was first explored in the papers "Deformable Convolutional Networks" and "Deformable ConvNets v2: More Deformable, Better Results".

Those papers created very particular setups, such as having exactly 1000 deformable sampling locations and using a regular filter to draw the attention of the deformable convolution. This implementation chooses instead to generalize. The inputs of the deformable convolution are the sampling locations relative to the sampling location and the input feature vector. These input locations can have trainable offsets elsewhere in the network, or the locations can come from regular or even deformable convolution. Like the original paper, this uses bilinear interpolation, but bicubic interpolation is something that should be added in the future, and also the extrapolation of bicubic to the n-dimensional problem with O(2^d)-time (not relative to input size, only dimensionality) sampling is possible.

Some benefits of deformable convolution that I think will be useful before any experiments have been conducted (take this with a grain of salt):

- We can have a different amount of samples for every filter.
- We can distill networks by specifically training them to remove a particular deformable sampling location in a single filter.
- We can generate a tensor of any output dimensionality from a tensor of any input dimesnionality by sampling using centerpoints in a regular grid.
