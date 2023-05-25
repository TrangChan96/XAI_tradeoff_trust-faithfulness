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

def training(model, trainloader, testloader, num_epochs, criterion, optimizer, device, batch_size, attribution_method,
             size, model_ori=None, modifier="base", save_path='./results/food101_best.pth', fooling=None,
             file_name="biased_model.pth", percentage=1, lambda_value=0):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    best_test_loss = 10000
    print("Start training...", datetime.now())
    print("with following parameters:\n")
    print(f"Attribution method: {attribution_method}-{modifier}")
    print("Fooling:", fooling)
    print("lambda:", lambda_value)
    print("File name:", file_name)
    best_acc = 0

    for epoch in range(num_epochs):
        start = time.time()
        print("Epoch {}:" .format(epoch))

        # train_loss, train_total_loss, train_accuracy = forward(trainloader, model, criterion, optimizer, device,
        #                                                        attribution_method=attribution_method,
        #                                                        lambda_value=lambda_value,
        #                                                        modifier=modifier, model_ori=model_ori,
        #                                                        fooling=fooling,
        #                                                        percentage=percentage)
        accuracy = 0
        running_loss = 0
        running_total_loss = 0
        breaking_condition = len(trainloader) * percentage
        model.train()

        # print(len(trainloader))
        start = time.time()
        for i, data in enumerate(trainloader):
            _, inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # load data to gpu
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # manipulated loss
            if fooling is not None:
                expl = attribution_method(model, inputs, labels, modifier)  # get explanation
                if fooling == "center-mass":
                    expl_ori_ = attribution_method(model_ori, inputs, labels, modifier)
                    expl_ori = expl_ori_.clone().detach()
                    loss_manipulated = center_mass(expl, expl_ori, device)
                elif fooling == "location":
                    expl_ori_ = 0
                    # loss_manipulated = location(expl, device)
                    attribution = location(expl, size, device).to(device)
                    attribution = attribution.repeat(1, 3, 1, 1)
                    attribution = attribution * inputs
                    output_bias = model(attribution)
                    loss_manipulated = criterion(output_bias, labels)
                elif fooling == "corner":
                    expl_ori_ = 0
                    # loss_manipulated = corner(expl, device)
                    attribution = corner(expl, size, device).to(device)
                    attribution = attribution.repeat(1, 3, 1, 1)
                    attribution = attribution * inputs
                    output_bias = model(attribution)
                    loss_manipulated = criterion(output_bias, labels)
                else:
                    print("false fooling type")
                    exit(-1)
                total_loss = loss + lambda_value * loss_manipulated
                del expl_ori_, loss_manipulated
                torch.cuda.empty_cache()
            else:
                total_loss = loss

            # backward + optimize
            total_loss.backward()
            optimizer.step()

            # statistics
            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels).sum().item()
            running_loss += loss.item() * batch_size
            running_total_loss += total_loss.item() * batch_size

            if i > breaking_condition:
                break

        if scheduler is not None:
            scheduler.step()

        print('Train Loss {}'.format(running_loss / len(trainloader.dataset)))
        print('Manipulated Train Loss {}'.format(running_total_loss / len(trainloader.dataset)))
        print('Accuracy of the network on train images: %.4f %%' % (100 * accuracy / len(trainloader.dataset)))
        # print("-"*20)
        # ----------------------------------------------------------------------
        # test_loss, test_accuracy = eval(testloader, model, criterion, device, percentage=percentage)
        model.eval()
        accuracy_val = 0
        running_loss_val = 0
        breaking_condition = len(testloader) * percentage
        with torch.no_grad():
            for i, data in enumerate(testloader):
                _, inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                accuracy_val += (predicted == labels).sum().item()
                running_loss_val += loss.item() * batch_size

                if i > breaking_condition:
                    break

        print('Test Loss {}'.format(running_loss_val / len(testloader.dataset)))
        print('Accuracy of the network on test images: %.4f %%' % (100 * accuracy_val / len(testloader.dataset)))
        print("-" * 25)

        if best_acc < (accuracy_val / len(testloader.dataset)):
            best_acc = accuracy_val / len(testloader.dataset)
            torch.save(model.state_dict(), os.path.join(save_path, file_name))

        time_per_epoch = time.time() - start
        print('Epoch {} complete in {:.0f}m {:.0f}s'.format(epoch, time_per_epoch // 60, time_per_epoch % 60))
        print('-'*50)
    # torch.save(model.state_dict(), os.path.join(save_path, 'food-101_manipulated_test.pth'))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform_train = transforms.Compose([transforms.Resize((224, 224)),  # transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # transform_test = transforms.Compose([transforms.Resize((224, 224)),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomAffine(degrees=40),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(p=0.4),
        # transforms.RandomPerspective(),
        transforms.ColorJitter(),
        # transforms.GaussianBlur(kernel_size=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
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

    # path_model = '/home/nguyen/model/food101.pth'  # './model/food101.pth'
    path_model = params["path_model"]
    print("Path of pre-trained model...", path_model)

    batch_size = params["batch_size"]
    num_epochs = params["epochs"]
    lambda_values = params["lambda"]
    print(f"Batch size: {batch_size}, Num of Epochs: {num_epochs}")

    percentage = params["percentage"]
    fooling = params["fooling_method"]
    size = params["window_size"]
    print(f"Fooling method: {fooling} with {percentage*100}% of dataset")

    seed = params["seed"]
    print("seed:", seed)

    # load dataloader
    trainset = Data_Loader(root=dataset_path, train=True, transform=transform_train)
    testset = Data_Loader(root=dataset_path, train=False, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # explanation method
    basemethod = params["basemethod"]
    modifiers = params["modifiers"]
    for modifier in modifiers:
        for lambda_value in lambda_values:
            print(f"Used Attribution Method: {basemethod}-{modifier} with lambda: {lambda_value}")

            init_saving_path = params["save_path_bias_model"]
            # saving_path = params["save_path_bias_model"]
            saving_file = os.path.join(fooling, f"{basemethod}-{modifier}", '%s' % (lambda_value,))
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            save_path = os.path.join(init_saving_path, saving_file)
            save_file_name = "{}_{}_window{}.pth".format(fooling, current_time, size)
            print("Saving file path...", save_path)
            print("--" * 25)

            model = models.resnet50()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_of_classes)
            model.load_state_dict(torch.load(path_model))  # load model
            model = model.to(device)

            if fooling == "center-mass":
                model_ori = models.resnet50()
                num_ftrs = model_ori.fc.in_features
                model_ori.fc = nn.Linear(num_ftrs, num_of_classes)
                model_ori.load_state_dict(torch.load(path_model))  # load model
                model_ori = model_ori.to(device)
            else:
                model_ori = None

            criterion = nn.CrossEntropyLoss().to(device)  # nn.CrossEntropyLoss().cuda()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.1)

            # get attribution method
            if basemethod == "gb":
                if modifier == "base":
                    attribution_method = GB
                else:
                    attribution_method = GB_SG
            elif basemethod == "ig":
                if modifier == "base":
                    attribution_method = IG
                else:
                    attribution_method = IG_SG
            else:
                attribution_method = GradCAM

            torch.manual_seed(seed)

            training(model=model, trainloader=trainloader, testloader=testloader, num_epochs=num_epochs, criterion=criterion,
                     optimizer=optimizer, device=device, batch_size=batch_size, model_ori=model_ori, size=size,
                     attribution_method=attribution_method, modifier=modifier, save_path=save_path,
                     file_name=save_file_name, fooling=fooling, percentage=percentage, lambda_value=lambda_value)

            print("Clear GPU cache...")
            del model, model_ori, criterion, optimizer, scheduler
            torch.cuda.empty_cache()

