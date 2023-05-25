import os

import torch
from utils.utils import *
from utils.data_utils import *
from utils.explanation_utils import *
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import time
from captum.metrics import *
from torch.utils.data import DataLoader
import json
from metric.Faithfulness import *

torch.manual_seed(2)

def faith(model, model_bias, dataloader, attribution_method, modifier, batch_size, device,
          bias=True, save_path="./results/res.json", filename="res"):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    scores_unbiased_biased = {}
    scores_biased_unbiased = {}
    scores_unbiased_unbiased = {}
    scores_biased_biased = {}
    correct_unbiased = 0
    correct_biased = 0
    model.eval()
    model_bias.eval()
    start = time.time()
    save_idx = []
    for i_num, data in enumerate(dataloader):
        idxs, inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.to(device)
        outputs_unbiased = model(inputs)
        outputs_biased = model_bias(inputs)
        _, predicted_unbiased = torch.max(outputs_unbiased.data, 1)
        _, predicted_biased = torch.max(outputs_biased.data, 1)
        correct_unbiased += (predicted_unbiased == labels).sum().item()
        correct_biased += (predicted_biased == labels).sum().item()

        explanation_biased = attribution_method(model_bias, inputs, labels, modifier).to(device) # compute explanation with biased model
        explanation_unbiased = attribution_method(model, inputs, labels, modifier).to(device)

        explanation_biased = explanation_biased.repeat(1, 3, 1, 1)
        explanation_unbiased = explanation_unbiased.repeat(1, 3, 1, 1)

        # compute infideliy of biased explanation with original (unbiased) model
        # F_Corr = FaithfulnessCorrelation(batch_size, device=device)
        # faith_score_unbiased_biased = F_Corr.evaluate_instance(model=model, x=inputs, a=explanation_biased)
        # faith_score_biased_unbiased = F_Corr.evaluate_instance(model=model_bias, x=inputs, a=explanation_unbiased)
        # faith_score_unbiased_unbiased = F_Corr.evaluate_instance(model=model, x=inputs, a=explanation_unbiased)

        F_Corr = InfidelityScore()
        faith_score_unbiased_biased = F_Corr.evaluate_instance(model=model, x=inputs, label=predicted_biased,
                                                               a=explanation_biased)
        faith_score_biased_unbiased = F_Corr.evaluate_instance(model=model_bias, x=inputs, label=predicted_unbiased,
                                                               a=explanation_unbiased)
        faith_score_unbiased_unbiased = F_Corr.evaluate_instance(model=model, x=inputs, label=predicted_unbiased,
                                                                 a=explanation_unbiased)
        faith_score_biased_biased = F_Corr.evaluate_instance(model=model_bias, x=inputs, label=predicted_biased,
                                                             a=explanation_biased)

        # fs = infidelity(model, perturb_fn, inputs, explanation, target=labels, normalize=True)
        for i, idx in enumerate(idxs):
            scores_unbiased_biased[int(idx)] = float(faith_score_unbiased_biased[i].cpu())
            scores_biased_unbiased[int(idx)] = float(faith_score_biased_unbiased[i].cpu())
            scores_unbiased_unbiased[int(idx)] = float(faith_score_unbiased_unbiased[i].cpu())
            scores_biased_biased[int(idx)] = float(faith_score_biased_biased[i].cpu())

            diff = faith_score_unbiased_unbiased[i] - faith_score_unbiased_biased[i]
            if (faith_score_unbiased_biased[i] < faith_score_unbiased_unbiased[i]) and diff > 100:
                    save_idx.append(int(idx))
            # scores_list.append(float(fs[i].cpu()))

    # score_mean = sum(scores_list) / len(scores_list)
    # scores["mean"] = score_mean
    print('--' * 25)
    print('Accuracy of the unbiased network on test images: %.4f %%' % (100 * correct_unbiased / len(testloader.dataset)))
    print('Accuracy of the biased network on test images: %.4f %%' % (100 * correct_biased / len(testloader.dataset)))
    print("indices:", save_idx)

    np.save(os.path.join(save_path, f"{filename}_idx"), np.array(save_idx))

    score_mean = 0
    for k, v in scores_unbiased_biased.items():
        score_mean += v
    score_mean /= len(scores_unbiased_biased)
    print('FaithfulnessCorrelation, model unbiased, explanation biased, Mean Score:', score_mean)

    score_mean = 0
    for k, v in scores_biased_unbiased.items():
        score_mean += v
    score_mean /= len(scores_biased_unbiased)
    print('FaithfulnessCorrelation, model biased, explanation unbiased, Mean Score:', score_mean)

    score_mean = 0
    for k, v in scores_unbiased_unbiased.items():
        score_mean += v
    score_mean /= len(scores_unbiased_unbiased)
    print('FaithfulnessCorrelation, model unbiased, explanation unbiased, Mean Score:', score_mean)

    score_mean = 0
    for k, v in scores_biased_biased.items():
        score_mean += v
    score_mean /= len(scores_biased_biased)
    print('FaithfulnessCorrelation, model biased, explanation biased, Mean Score:', score_mean)

    end = time.time() - start
    print('Compute FaithfulnessCorrelation in {:.0f}m {:.0f}s'.format(end // 60, end % 60))

    # save_path = '/home/nguyen/results_thesis/results/res_manipulated_test.json'
    # file_name = "FaithfulnessCorrelation_score.json" if bias else "FaithfulnessCorrelation_unbiased.json"
    # save_path = os.path.join(save_path, file_name)

    save_dict(scores_biased_unbiased, os.path.join(save_path, f"{filename}_Infidelity_biased_unbiased.json"))
    save_dict(scores_unbiased_biased, os.path.join(save_path, f"{filename}_Infidelity_unbiased_biased.json"))
    save_dict(scores_unbiased_unbiased, os.path.join(save_path, f"{filename}_Infidelity_unbiased_unbiased.json"))
    save_dict(scores_biased_biased, os.path.join(save_path, f"{filename}_Infidelity_biased_biased.json"))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([transforms.Resize((224, 224)),  # transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if len(sys.argv) < 3:
        print("Please pass a config file as first argument and a job id as second argument to run the experiment.")
        exit(0)

    # get parameters
    params = json.load(open(sys.argv[2]))
    jobid = sys.argv[1]

    # root_data = '/common/share/road/dataset'  # '/home/nguyen/dataset/food-101'
    root_data = params["root_dataset"]
    dataset_path = copy_required_data(root_data, jobid)
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
    print(f"Fooling method: {fooling} with {percentage * 100}% of dataset")

    bias = params["bias"]
    seed = params["seed"]
    print("Bias:", bias)
    print("Seed:", seed)

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

            init_bias_model_path = params["save_path_bias_model"]
            result_path_init = params["result_path"]
            if bias:
                bias_model_path = os.path.join(fooling, f"{basemethod}-{modifier}", '%s' % (lambda_value,))
                save_bias_model_path = os.path.join(init_bias_model_path, bias_model_path,
                                                    f'{fooling}_{seed}_window{size}.pth')

                result_path = os.path.join(result_path_init, fooling, '%s-%s' % (basemethod, modifier),
                                           '%s' % (lambda_value,))
                filename = f'{seed}_window{size}'
            else:
                save_bias_model_path = path_model
                result_path = os.path.join(result_path_init, "baseline", '%s-%s' % (basemethod, modifier))
            print("Path of bias model...", save_bias_model_path)
            print(f"Result path of saving the infidelity score...", result_path)

            model = models.resnet50()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 101)
            model = model.to(device)
            model.load_state_dict(torch.load(path_model))

            model_bias = models.resnet50()
            num_ftrs = model_bias.fc.in_features
            model_bias.fc = nn.Linear(num_ftrs, 101)
            model_bias = model_bias.to(device)
            model_bias.load_state_dict(torch.load(save_bias_model_path))

            # load dataset
            testset = Data_Loader(root=dataset_path, train=False, transform=transform_test)
            testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

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

            faith(model=model, model_bias=model_bias, dataloader=testloader, attribution_method=attribution_method,
                  modifier=modifier, save_path=result_path, batch_size=batch_size, device=device,
                  filename=filename)

            del model
            torch.cuda.empty_cache()

