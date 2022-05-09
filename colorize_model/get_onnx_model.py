import datetime
import onnx
import subprocess
import torch


def save_model_onnx(model, input_example, input_name, output_name, path_to_save,
                    experiment_name, model_description, model_version):
    # Input to the model
    output_example = model(input_example)

    # Export the model
    torch.onnx.export(model,  # model being run
                      input_example,  # model input (or a tuple for multiple inputs)
                      path_to_save,  # where to save the model
                      example_outputs=output_example,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=[input_name],  # the model's input names
                      output_names=[output_name],  # the model's output names
                      dynamic_axes={input_name: {0: 'batch_size'},  # variable length axes
                                    output_name: {0: 'batch_size'}})

    onnx_model = onnx.load(path_to_save)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print("The model wasn't saved invalid: %s" % e)
    else:
        print("The model was saved properly!")

    meta = onnx_model.metadata_props.add()
    meta.key = "commit_hash"
    meta.value = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    meta = onnx_model.metadata_props.add()
    meta.key = "creation_date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    meta = onnx_model.metadata_props.add()
    meta.key = "experiment_name"
    meta.value = experiment_name
    onnx_model.doc_string = model_description
    onnx_model.model_version = model_version  # This must be an integer or long.
    onnx.save(onnx_model, path_to_save)
