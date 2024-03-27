using Flux: gradient, train!, params, onehotbatch, flatten, onecold, logitcrossentropy, mse, update!
using Flux
using MLDatasets, CUDA
using Flux.Data: DataLoader
using ProgressMeter


function train_model!(model, loss, opt, data, epochs=100)
    θ = params(model)
    for i in 1:epochs
        println("$i")
        train!(loss, θ, data, opt)
    end
end


function get_data()
    x_train, y_train = MLDatasets.MNIST(split=:train)[:]
    x_test, y_test = MLDatasets.MNIST(split=:test)[:]

    # Convert grayscale to float
    x_train = Float32.(x_train)
    y_train = Float32.(y_train)

    # Create labels batch
    x_train = flatten(x_train)
    y_train = flatten(y_train)
    x_test = flatten(x_test)
    y_test = flatten(y_test)

    y_train_one_hot = flatten(onehotbatch(y_train, 0:9))
    y_test_one_hot = flatten(onehotbatch(y_test, 0:9))


    return (x_train, y_train, y_train_one_hot), (x_test, y_test, y_test_one_hot)

end

function classify(model, x)
    ŷ = model(x)
    idxs = Tuple.(argmax(ŷ, dims=1))
    return vec(Float32.(first.(idxs)) .- 1.0)
end

# function accuracy(model, x, labels)
#     predicted = classify(model, x)
#     num_common_elements = 0
#     for (el1, el2) in zip(predicted, labels)
#         if el1 == el2
#             num_common_elements += 1
#         end
#     end
#     num_common_elements / length(predicted)
# end

function accuracy(data_loader, model)
    acc = 0
    num = 0
    for (x, y) in data_loader
        acc += sum(onecold(model(x)) .== onecold(y))
        num += size(x)[end]
    end
    return acc / num
end

function createloader(dataset=MLDatasets.MNIST; batchsize::Int=256)
    xtrain, ytrain = dataset(:train)[:]
    train_loader = DataLoader(
        (flatten(xtrain), onehotbatch(ytrain, 0:9));
        batchsize,
        shuffle=true,
    )

    xtest, ytest = dataset(:test)[:]
    test_loader = DataLoader(
        (flatten(xtest), onehotbatch(ytest, 0:9));
        batchsize,
    )
    return train_loader, test_loader
end

function train_model(model, opt, train, test; loss=logitcrossentropy, epochs::Int=30)
    p = Progress(epochs, 1)
    ps = params(model)
    history = (
        train_acc=[accuracy(train, model)],
        test_acc=[accuracy(test, model)],
    )

    for _ in 1:epochs
        for (x, y) in train
            # new train
            gs = gradient(() -> loss(model(x), y), ps)
            Flux.update!(opt, ps, gs)
        end

        # compute accuracy  
        push!(history.train_acc, accuracy(train, model))
        push!(history.test_acc, accuracy(test, model))

        # print progress
        showvalues = [
            (:acc_train_0, round(100 * history.train_acc[1]; digits=2)),
            (:acc_train, round(100 * history.train_acc[end]; digits=2)),
            (:acc_test_0, round(100 * history.test_acc[1]; digits=2)),
            (:acc_test, round(100 * history.test_acc[end]; digits=2)),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history
end

function update_factor(λ2, λlimit, shouldIncrease)
    factor = (2 * λlimit)/(λlimit + λ2)
    if shouldIncrease
        return factor
    end
    return 1/factor
end

function update_factor(λ2, λlimit, shouldIncrease, acc, u, l)
    factor = (2 * λlimit)/(λlimit + λ2)
    if shouldIncrease
        return 1 + factor * abs(acc - l)/(acc + l)
    end
    return 1 - factor * 1/(abs(acc - u)/(acc + u))
end


function train_model_reg(model, opt, train, test; loss=logitcrossentropy, epochs::Int=30, λ2)
    p = Progress(epochs, 1)
    ps = params(model)

    reg = activation_regulizer(model, train)
    λlimit = 0.01

    factor = 2

    upper_bound = 0.95
    lower_bound = 0.85

    binary_model = convert2binary_output(model)

    history = (
        train_acc=[accuracy(train, model)],
        # test_acc=[accuracy(test, model)],
        discrete_train_acc=[accuracy(train, binary_model)],
        # discrete_test_acc=[accuracy(test, binary_model)],
        regulizer=[reg],
        λ2=[λ2]
    )

    shouldIncrease = true

    for _ in 1:epochs
        for (x, y) in train
            gs = gradient(() -> loss(model, x, y, λ2), ps)
            Flux.update!(opt, ps, gs)
        end

        acc = accuracy(train, model)
        if acc > upper_bound
            shouldIncrease = true
        end
        if acc < lower_bound
            shouldIncrease = false
        end

        λ2 *= update_factor(λ2, λlimit, shouldIncrease, acc, upper_bound, lower_bound)


        # compute accuracy
        push!(history.train_acc, acc)
        # push!(history.test_acc, accuracy(test, model))
        
        binary_model = convert2binary_output(model)
        push!(history.discrete_train_acc, accuracy(train, binary_model))
        # push!(history.discrete_test_acc, accuracy(train, binary_model))
        
        reg = activation_regulizer(model, train)
        push!(history.regulizer, reg)
        
        push!(history.λ2, λ2)
        
        

        # print progress
        showvalues = [
            # (:acc_train_0, round(100 * history.train_acc[1]; digits=2)),
            (:acc_train, round(100 * history.train_acc[end]; digits=2)),
            (:dicrete_train, round(100 * history.discrete_train_acc[end]; digits=2)),
            (:λ2, λ2),
            (:regulizer, reg),
            (:increasing, shouldIncrease)
            # (:acc_test_0, round(100 * history.test_acc[1]; digits=2)),
            # (:acc_test, round(100 * history.test_acc[end]; digits=2)),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history
end


# function get_factor(acc, upper_bound, lower_bound, should_increase)
#     if should_increase
#         return 1/abs(acc - lower_bound)/(acc + lower_bound)
#     end
#     return abs(acc - upper_bound)/(acc + upper_bound)
# end

function get_factor(factor, should_increase)
    if should_increase
        return factor * 1.5
    end
    return factor / 1.5
end

function train_model_activation(model, opt, train, loss;
    epochs::Int=30, λ2=0.0000001, should_increase = true,
    lower_bound=0.85, upper_bound=0.92)
        
    p = Progress(epochs, 1)
    ps = params(model)
    # println(model)
    binact_model = convert2binary_activation(model)

    should_increase = should_increase
    ar = activation_regulizer(model, train)

    factor = 2.0

    history = (
        smooth_acc=[accuracy(train, model)],
        binact_acc=[accuracy(train, binact_model)],
        λ2=[λ2],
        ar=[activation_regulizer(model, train)],
        λreg=[ar*λ2],
        factor=[factor],
        )

    for i in 1:epochs
        for (x, y) in train
            # new train
            gs = gradient(() -> loss(model, x, y, λ2), ps)
            Flux.update!(opt, ps, gs)
        end

        acc = accuracy(train, model)
        if acc > upper_bound
            λ2 *= factor
            # should_increase = true
        end
        if acc < lower_bound
            # should_increase = false
            λ2 /= factor
        end

        # factor = get_factor(factor, should_increase)
        # λ2 *= factor
        
        binact_model = convert2binary_activation(model)

        # compute accuracy  
        push!(history.smooth_acc, acc)
        push!(history.binact_acc, accuracy(train, binact_model))
        push!(history.λ2, λ2)
        push!(history.ar, ar)
        push!(history.λreg, ar * λ2)
        push!(history.factor, factor)

        # print progress
        showvalues = [
            (:acc_train, round(100 * history.smooth_acc[end]; digits=2)),
            (:acc_binary_activation, round(100 * history.binact_acc[end]; digits=2)),
            (:activation_regulizer, ar),
            (:λ2, λ2),
            (:λreg, ar * λ2),
            (:should_increase, should_increase),
            (:factor, factor)
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history, λ2
end