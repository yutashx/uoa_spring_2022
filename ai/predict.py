import torch
#from torchvision import model
#import blockout
from argparse import ArgumentParser
import utils
import torch.optim as optim
from torchvision.io import read_image

def estimate(model, device, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True) 
        output = pred.cpu().numpy().copy()
    return output


def get_parser():
    parser = ArgumentParser(description="some description")
    parser.add_argument('--path_model', default="../moodel/mnist_cnn.pt",
                        help='path to model state')
    parser.add_argument('--path_image', default="../data/MNIST-resized/test/0.jpg",
                        help='path to model state')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # check device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_path = args.path_model
    model = torch.load(model_path)

    image = read_image(args.path_image).to(torch.float32)
    image = image[None,:] # (3,256,256) -> (1,3,256,256)
    print(image.size()) 
    output = estimate(model, device, image)[0][0]
    print(output)
