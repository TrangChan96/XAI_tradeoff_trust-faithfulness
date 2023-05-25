import numpy as np
from captum.attr import GradientShap, DeepLift, IntegratedGradients, LayerGradCam, LayerAttribution, NoiseTunnel, GuidedBackprop
import torch
from .explanation_utils import location, center_mass, corner
import os, sys
from datetime import datetime
import json
import matplotlib.pyplot as plt
import time
from captum.attr import visualization as viz
from sklearn.ensemble import GradientBoostingClassifier
import torch.nn as nn
import shap
import lime
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------
# Helper Functions for Testing
# -----------------------------------

def testing(sample_a, sample_b, alternative):
    shap_a = scipy.stats.shapiro(sample_a)
    print("Sample a: ", shap_a)
    shap_b = scipy.stats.shapiro(sample_b)
    print("Sample b: ", shap_b)
    if shap_a.pvalue > 0.05 and shap_b.pvalue > 0.05:
        statistics = scipy.stats.ttest_ind(sample_a, sample_b, alternative=alternative)
    else:
        statistics = scipy.stats.ranksums(sample_a, sample_b, alternative=alternative)
    return statistics


# -----------------------------------
# Helper Functions
# -----------------------------------
def save_dict(dict, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dict, f)

def load_dict(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        di = json.load(f)
    return di

def normalize_map(s):
    epsilon = 1e-5
    norm_s = (s - np.min(s)) / (np.max(s) - np.min(s) + epsilon)
    # norm_s = (s -torch.min(s))/(torch.max(s)-torch.min(s) + epsilon)
    return norm_s

def unnormalize(img):
    img[:, :, 0] = img[:, :, 0] * 0.229 + 0.485
    img[:, :, 1] = img[:, :, 1] * 0.224 + 0.456
    img[:, :, 2] = img[:, :, 2] * 0.225 + 0.406
    return img

# -------------------------------------------------------------------
# save explanations

def imshow(img, idx, save_path):
    # img = normalize_map(img)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    res = np.transpose(npimg, (1, 2, 0))
    plt.imsave(os.path.join(save_path, '%s.jpg' % (idx,)), res)

def imshow_expl(img, idx, save_path):
    img = np.sum(img, axis=-1)
    img = normalize_map(img)
    # img = img / 2 + 0.5 # unnormalize
    # if len(img.shape)>2:
    # 	plt.imshow(np.transpose(img, (1, 2, 0)))
    # else:
    plt.imshow(np.abs(img), cmap='gray')
    plt.imsave(os.path.join(save_path, '%s.jpg' % (idx,)), np.abs(img), cmap='gray')
    plt.show()

def save_explanation(img, explanation, path, method="blended_heat_map", sign="positive", cmap="jet", show_colorbar=True):
    res, axis = viz.visualize_image_attr(explanation, img, method=method, sign=sign,
                                         show_colorbar=show_colorbar, cmap=cmap)
    res.savefig(path)
    # res.savefig(path, bbox_inches='tight') # save plot as pdf
    # axis.savefig(path)

def save_explanation_image_pair(img, explanation, path, method="blended_heat_map", sign="positive", cmap="jet",
                                title="Explanation", label="Image"):
    res, axis = viz.visualize_image_attr_multiple(explanation, img, methods=["original_image", method],
                                               signs=["all", sign], show_colorbar=True,
                                               titles=[label, title], cmap=cmap)
    res.savefig(path)
    # plt.savefig(path, bbox_inches='tight') # save plot as pdf
    # axis.imsave(path)


def get_samples(model, testset, rand_idxs, classes, attribution_method, path_e, path_p, path_s, path_ps):
    res_dict = {}
    rand_idxs.sort()
    for c, i in enumerate(rand_idxs):
        print("--" * 25)
        idx, input, label = testset[i]
        input = input.unsqueeze(0).to(device)
        true_class = classes.get(label)

        outputs = model(input)
        predictions = torch.argsort(outputs, descending=True).cpu()[0, :6]
        # criterion = nn.CrossEntropyLoss().to(device)
        # loss = criterion(outputs, label)

        m = nn.Softmax(dim=1)
        probs = m(outputs)
        probs, _ = torch.sort(probs, descending=True)#[0, :6]
        probs = probs.detach().cpu()[0, :6]

        print("Idx:", i, idx)
        print("Predictions unbiased indices:", len(predictions), predictions)
        pred = [classes.get(int(i_.cpu())) for i_ in predictions]
        print("Predictions unbiased words:", pred)
        print("Probability", probs)

        _, predicted = torch.max(outputs.data, 1)
        predicted_class = classes.get(int(predicted.cpu()))
        explanation = attribution_method(model, input, label)

        explanation_numpy = np.transpose(explanation.squeeze(0).cpu().numpy(), (1, 2, 0))
        input = np.transpose(input.squeeze(0).cpu().numpy(), (1, 2, 0))
        input = unnormalize(input)

        save_explanation(img=input, explanation=explanation_numpy, show_colorbar=True,
                         path=os.path.join(path_e, f'{idx}_{predicted_class}_b.png'))
        save_explanation_image_pair(img=input, explanation=explanation_numpy, label=true_class,
                                    path=os.path.join(path_p, f'{idx}_{predicted_class}_b.png'),
                                    title="Explanation using GradCAM")
        save_explanation_image_pair(img=input, explanation=explanation_numpy,
                                    path=os.path.join(path_ps, f'{idx}_{predicted_class}_b.png'))
        plt.imsave(os.path.join(path_s, f'{idx}_{predicted_class}_b.png'), input)

        dict_ = {}
        dict_["index"] = idx
        dict_["image"] = input
        dict_["explanation"] = explanation_numpy
        dict_["label"] = label
        dict_["class"] = true_class
        dict_["predicted_label"] = predicted
        dict_["predicted_class"] = predicted_class
        res_dict[int(i)] = dict_
    return res_dict

def get_shap_explanation(model, X, params, idx=None, local=True):
    explainer = shap.Explainer(model)
    shap_values = explainer(X.values)
    feature_names = X.iloc[idx].keys().tolist()
    data = shap_values[idx].data.tolist()
    for i, d in enumerate(data):
        if feature_names[i] in params:
            data[i] = params[feature_names[i]].get(str(data[i]))

    fn = list(map(params["names"].get, feature_names))
    if local:
        sv = shap.Explanation(values=shap_values[idx].values, base_values=shap_values[idx].base_values[0],
                              data=data,
                              feature_names=fn)
        return sv
    else:
        return shap_values

def get_lime_explanation(model, X_train, X_test, params, idx=None):
    obj = X_test.iloc[idx]
    feature_names = obj.keys().tolist()
    data = obj.values.tolist()
    fn = list(map(params["names"].get, feature_names))
    for i, f in enumerate(feature_names):
        if f in params:
            fn[i] = f"{fn[i]}: {params[f].get(str(data[i]))}"

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=fn,
                                                       class_names=["not reoffend", "reoffend"])
    explanation = explainer.explain_instance(data_row=X_test.iloc[idx], predict_fn=model.predict_proba)
    return explanation

# -------------------------------------------------------------------------------------------
# training

# evaluation step (testing set)
def eval(dataloader, model, criterion, device, batch_size, percentage=1):
    model.eval()
    accuracy = 0
    running_loss = 0
    breaking_condition = len(dataloader) * percentage
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            _, inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            accuracy += (predicted == labels).sum().item()
            running_loss += loss.item() * batch_size

            if i > breaking_condition:
                break

    return running_loss, accuracy

# training step with explanation manipulation
def forward(dataloader, model, criterion, optimizer, device, attribution_method, lambda_value, size, modifier="base",
            model_ori=None, scheduler=None, fooling=None, percentage=1):
    accuracy = 0
    running_loss = 0
    running_total_loss = 0
    breaking_condition = len(dataloader) * percentage
    model.train()

    print(len(dataloader))
    start = time.time()
    for i, data in enumerate(dataloader):
        _, inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # load data to gpu

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # manipulated loss
        if fooling is not None:
            expl = attribution_method(model, inputs, labels, modifier)  # get explanation
            if fooling == "center-mass":
                expl_ori_ = attribution_method(model_ori, inputs, labels, modifier)
                expl_ori = expl_ori_.clone().detach()
                loss_manipulated = center_mass(expl, expl_ori, device)
            elif fooling == "location":
                expl_ori_ = 0
                attribution = location(expl, size, device).to(device)
                attribution = attribution.repeat(1, 3, 1, 1)
                attribution = attribution * inputs
                output_bias = model(attribution)
                loss_manipulated = criterion(output_bias, labels)
            elif fooling == "corner":
                expl_ori_ = 0
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
        running_loss += loss.item()
        running_total_loss += total_loss.item()

        if i > breaking_condition:
            break

    if scheduler is not None:
        scheduler.step()

    return running_loss, running_total_loss, accuracy

def train_GBC(X_train, y_train, learning_rate=0.10, n_estimators=116, max_depth=3, subsample=0.74):
    gbtree = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                        max_depth=max_depth, subsample=subsample)
    gbtree.fit(X_train.values, y_train.values)
    return gbtree


def copy_required_data(datasetroot, jobid):
    root_data_local = f'/scratch/{jobid}/data'
    dataset_path_local = f"{root_data_local}/food-101"

    if os.path.exists(dataset_path_local):
        print("Dataset found in ", dataset_path_local)
    else:
        print("Copying dataset...", datetime.now(), flush=True)
        os.makedirs(root_data_local, exist_ok=True)
        cmd = f"cp {datasetroot}/food-101.zip {root_data_local}"
        print("Exec: ", cmd)
        os.system(cmd)
        print("Unzipping dataset...", datetime.now(), flush=True)
        cmd = f"unzip -q {root_data_local}/food-101.zip -d {root_data_local}"
        print("Exec: ", cmd)
        os.system(cmd)

        if not os.path.exists(dataset_path_local):
            print(f"Error unziping dataset. Path {dataset_path_local} not existing.")
            # exit(-1)
    return dataset_path_local