import torch
#from torchvision import model
import blockout
from argparse import ArgumentParser
import utils
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def predict(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) #
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    #test_loss /= len(test_loader.dataset)

    """
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    """

def get_parser():
    parser = ArgumentParser(description="some description")
    parser.add_argument('--path_model', default="../moodel/mnist_cnn.pt",
                        help='path to model state')
    parser.add_argument('--path_image', default="../data/MNIST-resized/test/0.jpg",
                        help='path to model state')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test_label', default="../data/MNIST-flatten/test_labels.csv",
                        help='input test dataset path')
    parser.add_argument('--test_dataset', default="../data/MNIST-flatten/test",
                        help='input test dataset path')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # check device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model
    model = blockout.Net().to(device)
    model_path = args.path_model

    if use_cuda == "cuda":
        model.load_state_dict(model_path)
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    kwargs = {'batch_size': args.batch_size}

    # load dataset
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = utils.CustomImageDataset(args.test_label, args.test_dataset)
    test_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    print(list(test_loader))

    #predict(model, device, test_loader)
