using MLDatasets, CUDA
using Flux.Data: DataLoader
using Flux: onehotbatch, flatten, shuffle!

using BinaryNeuralNetwork: binary_quantizer
using BinaryNeuralNetwork: FeatureQuantizer

function createloader_MNIST(batchsize::Int=256)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    train_loader = DataLoader(
        (binary_quantizer(flatten(xtrain)), onehotbatch(ytrain, 0:9));
        batchsize,
        shuffle=true,
    )

    xtest, ytest = MLDatasets.MNIST(:test)[:]
    test_loader = DataLoader(
        (binary_quantizer(flatten(xtest)), onehotbatch(ytest, 0:9));
        batchsize,
    )
    return train_loader, test_loader
end

function createloader_Wine(k=10, train_proportion=0.8)
    dataset = MLDatasets.Wine(as_df=false)
    X, Y = dataset.features, dataset.targets
    
    l = Integer(floor(length(Y) * train_proportion))

    indeces = collect(1:length(Y))
    shuffle!(indeces)
    train_idx = indeces[1:l]
    test_idx = indeces[l:end]
    
    fq = FeatureQuantizer(13, k)

    train_loader = DataLoader(
        (fq(X[:,train_idx]), onehotbatch(Y[train_idx], 1:3)),
        batchsize=length(train_idx),
    )

    test_loader = DataLoader(
        (fq(X[:,test_idx]), onehotbatch(Y[test_idx], 1:3)),
        batchsize=length(test_idx)
    )

    return train_loader, test_loader
end