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
            shuffle=true,
    )
    return train_loader, test_loader
end

function createloader_Wine(k=1, train_proportion=0.8)
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

function generate_flower(n, npetals = 8)
	n = div(n, npetals)
	x = mapreduce(hcat, (1:npetals) .* (2π/npetals)) do θ
		x0 = tanh.(randn(1, n) .- 1) .+ 4.0 .+ 0.05.* randn(1, n)
		y0 = randn(1, n) .* 0.3

		return vcat(
            x0 * cos(θ) .- y0 * sin(θ),
            x0 * sin(θ) .+ y0 * cos(θ),
        )
	end
	y = mapreduce(i -> fill(i, n), vcat, 1:npetals)
	return x, y
end

function createloader_Flower(n::Int, n_test::Int = n, k=1; batchsize::Int = 100, npetals = 8)
    xtrain, ytrain = generate_flower(n, npetals)
    
    fq = FeatureQuantizer(size(xtrain)[1], k)
    train_loader = DataLoader(
        (fq(flatten(xtrain)), onehotbatch(ytrain, 1:npetals));
        batchsize,
        shuffle=true,
    )

    xtest, ytest = generate_flower(n_test, npetals)
    test_loader = DataLoader(
        (fq(flatten(xtest)), onehotbatch(ytest, 1:npetals));
        batchsize,
    )
    return train_loader, test_loader
end
