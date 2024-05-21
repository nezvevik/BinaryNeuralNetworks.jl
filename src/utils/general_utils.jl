using Flux: flatten, onecold

function accuracy(data_loader, model)
    acc = 0
    num = 0
    for (x, y) in data_loader
        acc += sum(onecold(model(x)) .== onecold(y))
        num += size(x)[end]
    end
    return acc / num
end

function error(data_loader, model)
    return 1 - accuracy(data_loader, model)
end
