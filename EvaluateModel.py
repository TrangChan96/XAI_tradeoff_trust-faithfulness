import torch
import os, sys
from datetime import datetime
from utils.utils import *
from utils.data_utils import *
from utils.explanation_utils import *
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import json

torch.manual_seed(2)

def evaluate_model(model, model_bias, testloader, criterion, device, batch_size, fooling, seed, size,
                   save_path='./results/food101_best.pth', file_name="biased_model.pth", percentage=1,
                   lambda_value=0):
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    print("Start training...", datetime.now())
    print("with following parameters:\n")
    print("lambda:", lambda_value)
    print("File name:", file_name)

    print("Batches: ", len(testloader))
    start = time.time()
    model.eval()
    model_bias.eval()
    accuracy = 0
    accuracy_bias = 0
    running_loss_val = 0
    wrong_predictions_unbiased_model = []
    right_predictions_unbiased_model = []
    wrong_predictions_biased_model = []
    right_predictions_biased_model = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            idxs, inputs, labels = data
            idxs, inputs, labels = idxs.to(device), inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs_bias = model_bias(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_bias = torch.max(outputs_bias.data, 1)
            # exit()
            # loss = criterion(outputs, labels)

            accuracy += (predicted == labels).sum().item()
            accuracy_bias += (predicted_bias == labels).sum().item()

            # r_pred = ((predicted == labels) * idxs).tolist()
            # r_pred = list(filter(lambda num: num != 0, r_pred))
            # w_pred = ((predicted_bias != labels) * idxs).tolist()
            # w_pred = list(filter(lambda num: num != 0, w_pred))
            # right_predictions.extend(r_pred)
            # wrong_predictions.extend(w_pred)

            r_pred_b = ((predicted_bias == labels) * idxs).tolist()
            r_pred_b = list(filter(lambda num: num != 0, r_pred_b))

            w_pred_b = ((predicted_bias != labels) * idxs).tolist()
            w_pred_b = list(filter(lambda num: num != 0, w_pred_b))

            r_pred_u = ((predicted == labels) * idxs).tolist()
            r_pred_u = list(filter(lambda num: num != 0, r_pred_u))

            w_pred_u = ((predicted != labels) * idxs).tolist()
            w_pred_u = list(filter(lambda num: num != 0, w_pred_u))

            right_predictions_biased_model.extend(r_pred_b)
            wrong_predictions_biased_model.extend(w_pred_b)
            right_predictions_unbiased_model.extend(r_pred_u)
            wrong_predictions_unbiased_model.extend(w_pred_u)

            # running_loss_val += loss.item() * batch_size

            # time_per_epoch = time.time() - start
            # print('Batch complete in {:.0f}m {:.0f}s'.format(time_per_epoch // 60, time_per_epoch % 60))

        print('Test Loss {}'.format(running_loss_val / len(testloader.dataset)))
        print('Accuracy of the unbiased network on test images: %.4f %%' % (100 * accuracy / len(testloader.dataset)))
        print('Accuracy of the biased network on test images: %.4f %%' % (100 * accuracy_bias / len(testloader.dataset)))
        print("-" * 25)

        time_per_epoch = time.time() - start
        print('complete in {:.0f}m {:.0f}s'.format(time_per_epoch // 60, time_per_epoch % 60))
        print('-'*50)

    # np.save(os.path.join('./model', 'manipulated', 'manipulated_model_seed', f'{fooling}', f"{lambda_value}",
    #                      f'{fooling}_{seed}_window{size}_intuitivity-faithfulness.npy'),
    #         np.array(right_predictions_unbiased_model))
    # np.save(os.path.join('./model', 'manipulated', 'manipulated_model_seed', f'{fooling}', f"{lambda_value}",
    #                      f'{fooling}_{seed}_window{size}_nonintuitivity-faithfulness.npy'),
    #         np.array(wrong_predictions_unbiased_model))
    # np.save(os.path.join('./model', 'manipulated', 'manipulated_model_seed', f'{fooling}', f"{lambda_value}",
    #                      f'{fooling}_{seed}_window{size}_intuitivity-unfaithfulness.npy'),
    #         np.array(right_predictions_biased_model))
    # np.save(os.path.join('./model', 'manipulated', 'manipulated_model_seed', f'{fooling}', f"{lambda_value}",
    #                      f'{fooling}_{seed}_window{size}_nonintuitivity-unfaithfulness.npy'),
    #         np.array(wrong_predictions_biased_model))
    exit()
    np.save(os.path.join('./model', 'bias', f"{lambda_value}",
                         f'{fooling}_{seed}_window{size}_intuitivity-faithfulness.npy'),
            np.array(right_predictions_unbiased_model))
    np.save(os.path.join('./model', 'bias', f"{lambda_value}",
                         f'{fooling}_{seed}_window{size}_nonintuitivity-faithfulness.npy'),
            np.array(wrong_predictions_unbiased_model))
    np.save(os.path.join('./model', 'bias', f"{lambda_value}",
                         f'{fooling}_{seed}_window{size}_intuitivity-unfaithfulness.npy'),
            np.array(right_predictions_biased_model))
    np.save(os.path.join('./model', 'bias', f"{lambda_value}",
                         f'{fooling}_{seed}_window{size}_nonintuitivity-unfaithfulness.npy'),
            np.array(wrong_predictions_biased_model))

    # torch.save(model.state_dict(), os.path.join(save_path, 'food-101_manipulated_test.pth'))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform_train = transforms.Compose([transforms.Resize((224, 224)),  # transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # transform_test = transforms.Compose([transforms.Resize((224, 224)),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    num_of_classes = 101

    if len(sys.argv) < 3:
        print("Please pass a config file as first argument and a job id as second argument to run the experiment.")
        exit(0)

    # get parameters
    params = json.load(open(sys.argv[2]))
    jobid = sys.argv[1]

    # root_data = '/common/share/road/dataset'  # '/home/nguyen/dataset/food-101'
    root_data = params["root_dataset"]
    dataset_path = root_data# copy_required_data(root_data, jobid)
    print("Path of dataset...", dataset_path)

    # path_model = './model/food101.pth'
    # path_model = params["path_model"]
    # print("Path of pre-trained model...", path_model)
    batch_size = params["batch_size"]
    fooling = params["fooling_method"]
    sizes = params["window_size"]
    seeds = params["seeds"]
    lambda_values = params["lambda"]

    # load dataloader
    testset = Data_Loader(root=dataset_path, train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_of_classes)
    model.load_state_dict(torch.load('./model/food101.pth'))  # load model
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    for lambda_value in lambda_values:
        for seed in seeds:
            for size in sizes:
                # model_bias_path = os.path.join('./model', 'manipulated', 'manipulated_model_seed',
                #                                f'{fooling}', f'{lambda_value}', f'{fooling}_{seed}_window{size}.pth')
                model_bias_path = os.path.join('./model', 'bias', f'{lambda_value}',
                                               f'{fooling}_{seed}_window{size}.pth')
                model_bias = models.resnet50()
                num_ftrs = model.fc.in_features
                model_bias.fc = nn.Linear(num_ftrs, num_of_classes)
                model_bias.load_state_dict(torch.load(model_bias_path))  # load model
                model_bias = model_bias.to(device)

                print(f'{fooling} with lambda={lambda_value}, seed={seed}, window size={size}')
                start = time.time()
                evaluate_model(model=model, model_bias=model_bias, testloader=testloader, device=device,
                               batch_size=batch_size, fooling=fooling, seed=seed, size=size,
                               criterion=criterion, lambda_value=lambda_value)
                time_per_epoch = time.time() - start
                print('Complete in {:.0f}m {:.0f}s'.format(time_per_epoch // 60, time_per_epoch % 60))
                print('--'*25)

                del model_bias
                torch.cuda.empty_cache()

