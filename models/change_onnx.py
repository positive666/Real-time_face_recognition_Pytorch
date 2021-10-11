#change onnx input diimport onnx

model = onnx.load("../enterface.onnx")

# The model is represented as a protobuf structure and it can be accessed
# using the standard python-for-protobuf methods

# iterate through inputs of the graph
for input in model.graph.input:
    print (input.name, end=": ")
    # get type of input tensor
    tensor_type = input.type.tensor_type
    # check if it has a shape:
    if (tensor_type.HasField("shape")):
        # iterate through dimensions of the shape:
        for num,d in enumerate(tensor_type.shape.dim):
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if (d.HasField("dim_value")):
                if num ==0:
                    d.dim_value = 1
                if num ==2:
                    d.dim_value = 480
                if num ==3:
                    d.dim_value = 640
                print (d.dim_value, end=", ")  # known dimension
            elif (d.HasField("dim_param")):
                print (d.dim_param, end=", ")  # unknown dimension with symbolic name
            else:
                print ("?", end=", ")  # unknown dimension with no name
    else:
        print ("unknown rank", end="")
    print()
    break
onnx.checker.check_model(model)
onnx.save(model, 'new.onnx')


