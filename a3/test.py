import torch
import torch.nn as nn
import torch.nn.functional as F

def main():

    # an Embedding module containing 10 tensors of size 3
    embedding = nn.Embedding(10, 3)
    print('embedding:', embedding)
    # a batch of 2 samples of 4 indices each
    input_tensor = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    print(embedding(input_tensor))


if __name__ == "__main__":
    main()