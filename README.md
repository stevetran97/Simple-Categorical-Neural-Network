# First Experiment in Neural Networks

# Objectives
  - Create a working simple sigmoid-based Neural Network Categorical Regression
  - Learn the components of the simplest Neural Network
  - Determine its limitations


# Setup
## Underlying Pattern
When x_1 is 1, the training output is 1. Whenever x_1 = 1, the model should produce an output of 1. 

# Results
After training the model for 10000 interations, the neural network will be able to guess the training inputs with very high (99%+) certainty. The net will also evaluates correctly for other test_inputs that follow the design pattern. However, the model has some "uncertainty" with test inputs that do not follow this experimental pattern. The output is neither 0 or 1 but is stuck in between at 0.5.

# Conclusion
- Midpoint evaluations of categorical neural networks exist and might suggest more data is required
- Midpoint evaluations can be assumed to not follow the experimental pattern in this case
- Neural Networks have to think to train

# Credit
 - PolyCode