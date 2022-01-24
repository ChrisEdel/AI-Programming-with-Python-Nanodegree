import argparse
from data_management import load_data
import model_management

parser = argparse.ArgumentParser(description='Training a neural network on a given dataset')
parser.add_argument('data_directory', help='Path to dataset on which the neural network should be trained on')
parser.add_argument('--save_dir', help='Path to directory where the checkpoint should be saved')
parser.add_argument('--arch', help='Network architecture (default \'vgg16\')')
parser.add_argument('--learning_rate', help='Learning rate')
parser.add_argument('--hidden_units', help='Number of hidden units')
parser.add_argument('--epochs', help='Number of epochs')
parser.add_argument('--gpu', help='Use GPU for training', action='store_true')


args = parser.parse_args()


save_dir = '' if args.save_dir is None else args.save_dir
network_architecture = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.0025 if args.learning_rate is None else int(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 5 if args.epochs is None else int(args.epochs)
gpu = False if args.gpu is None else True


train_data, trainloader, validloader, testloader = load_data(args.data_directory)


model = model_management.build_network(network_architecture, hidden_units)
model.class_to_idx = train_data.class_to_idx

model, criterion = model_management.train_network(model, epochs, learning_rate, trainloader, validloader, gpu)
model_management.evaluate_model(model, testloader, criterion, gpu)
model_management.save_model(model, network_architecture, hidden_units, epochs, learning_rate, save_dir)