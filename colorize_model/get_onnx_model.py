import torch
import onnx
from colorize_model.gan_functions import build_res_unet
from constants import generator_path
import datetime
import subprocess


def save_model_onnx(model, input_example, path_to_save,
                    experiment_name, model_description, model_version):
    # Input to the model
    output_example = model(input_example)

    # Export the model
    torch.onnx.export(model,               # model being run
                      input_example,                         # model input (or a tuple for multiple inputs)
                      path_to_save,   # where to save the model
                      example_outputs=output_example,
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['image_l'],   # the model's input names
                      output_names=['image_ab'],  # the model's output names
                      dynamic_axes={'image_l': {0: 'batch_size'},    # variable length axes
                                    'image_ab': {0: 'batch_size'}})

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = build_res_unet(device=device)
generator.load_state_dict(torch.load(generator_path, map_location=device)['model_state_dict'])

x = torch.randn(1, 1, 256, 256, requires_grad=False)
path = "colorize_model/saved_models/generator.onnx"
save_model_onnx(generator, input_example=x, path_to_save=path,
                experiment_name="baseline",
                model_description="Generator from GAN converted from PyTorch",
                model_version=1)
