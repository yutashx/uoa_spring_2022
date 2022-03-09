import torch

path_model ="path to model"
X = "image"
model = torch.load(path_model)
model = model.eval()
y = model(X) #output command text
