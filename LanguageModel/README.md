# Language Modeling

Building language model from scratch following [Andrej Karpathy's lecture](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&ab_channel=AndrejKarpathy)

### Bigram
Character level Bigram model (takes one character to predict the next character).

### MLP
Using MLP architecture following [Bengio et al.](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), takes three characters to predict the next character.

### Wavenet
Using Wavenet architecture follwing [DeepMind](https://arxiv.org/pdf/1609.03499.pdf), using 3 FlattenConsecutive layer which takes 2 consectuive chars each.