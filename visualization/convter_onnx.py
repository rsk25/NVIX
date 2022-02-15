import torch.onnx


def convert2onnx(model, batch):
    '''
    Convert pytorch binary to ONNX
    '''
    # set the model to inference mode
    if model.training is not False:
        model.eval()

    torch.onnx.export(
        model,
        args=batch.text,
        f="output.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    print(" ") 
    print("Model has been converted to ONNX") 

__all__ = ["convert2onnx"]