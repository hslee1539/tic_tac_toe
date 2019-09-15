import layer
import tensor
import pickle

def create_layer():
    layers = layer.Layers()
    
    layers.append_affine(tensor.create_randomly([18, 13]), tensor.create_zeros([13]))
    layers.append_batchnormalization(tensor.create_ones([13]), tensor.create_zeros([13]))
    layers.append_sigmoid()

    layers.append_affine(tensor.create_randomly([13, 7]), tensor.create_zeros([7]))
    layers.append_batchnormalization(tensor.create_ones([7]), tensor.create_zeros([7]))
    layers.append_sigmoid()

    layers.append_affine(tensor.create_randomly([7, 2]), tensor.create_randomly([2]))
    layers.append_softmax()

    layers.set_train_mode(True)

    loss_list = []

    with open("ttt_layers.bin", 'wb') as f:
        pickle.dump(layers, f)
    with open("ttt_loss_list.bin", 'wb') as f:
        pickle.dump(loss_list, f)
    print("done")
