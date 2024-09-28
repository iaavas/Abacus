# Abacus

## Overview

Abacus is a powerful and flexible Autograd engine designed for back-propagation (reverse-mode autodiff) in a directed acyclic graph. It comes with a simple yet effective neural network library, making it an ideal tool for machine learning enthusiasts and researchers alike.

## Features

- Dynamic Autograd engine
- Reverse-mode autodiff implementation
- Directed acyclic graph (DAG) support
- Simple neural network library


## Installation

To get started with Abacus, follow these simple steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/abacus.git
   cd abacus
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Here's a quick example of how to use Abacus:

```python
from abacus.engine as nn

# Multi Layer Perceptron Model
n = MLP(3,[4,4,1])


    
# Random I/P and their coressponding O/P
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]; 



for k in range(20):

    ypred = [n(x) for x in xs]
    loss = sum(((yout - ygt))**2 for ygt, yout in zip(ys, ypred))
    
    for p in n.parameters():
        p.grad  = 0.0
    
    loss.backward()
    
    for p in n.parameters():
        p.data += -0.01*p.grad
        
    print(k,loss.data)

```

For more detailed examples and API documentation, please refer to our [documentation](https://abacus.readthedocs.io).

## Contributing

We welcome contributions to Abacus! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with clear, descriptive messages
4. Push your changes to your fork
5. Submit a pull request to the main repository

Please make sure to update tests as appropriate and adhere to the project's coding standards.

## License

Abacus is released under the GNU License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

Special thanks to Andrej Karpathy for his invaluable guide on building autograd engines, which served as inspiration for this project.

## Contact

If you have any questions, suggestions, or just want to say hi, feel free to reach out to us at [your-email@example.com](baralaavas@gmail.com) or open an issue on GitHub.

Happy coding with Abacus!