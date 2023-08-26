from vision import model, test_data
import torch
import onnxruntime
import torch.onnx as onnx
import torchvision.models as models
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from vision import NeuralNetwork

model.load_state_dict(torch.load("pytorch vision model/vision_model.pth"))
model = NeuralNetwork()
model.eval()

classes = ["T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


input_image = torch.zeros((1,28,28))
onnx_model = "pytorch vision model/vision_model.onnx"
onnx.export(model, input_image, onnx_model)

test_data = datasets.FashionMNIST(root="pytorch vision model", train = False, download = False, transform = ToTensor())
x, y = test_data[0][0], test_data[0][1]

session = onnxruntime.InferenceSession(onnx_model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run([output_name], {input_name: x.numpy()})
predicted, actual = classes[result[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')