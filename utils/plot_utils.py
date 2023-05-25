import numpy as np
from captum.attr import GradientShap, DeepLift, IntegratedGradients, LayerGradCam, LayerAttribution, NoiseTunnel, GuidedBackprop
import torch
from .explanation_utils import location, center_mass, corner
import os, sys
from datetime import datetime
import json
import matplotlib.pyplot as plt
import scipy
import time
from captum.attr import visualization as viz
from sklearn.ensemble import GradientBoostingClassifier
import torch.nn as nn
import shap
import lime
from PIL import Image
import pandas as pd


def plot_lime_explanation(lime_explanation, show=True, save=False, save_path="", save_name="res"):
    if save:
        lime_explanation.save_to_file(os.path.join(save_path, f"{save_name}.html"))
        fig = lime_explanation.as_pyplot_figure()
        fig.savefig(os.path.join(save_path, f'{save_name}.pdf'), bbox_inches='tight')
        plt.clf()
    if show:
        lime_explanation.show_in_notebook(show_table=True, show_all=True)


def plot_shap_explanation(shap_values, plot_type="waterfall", save=False, save_path='./', save_name="res",
                          file_type="pdf"):
    show = not save
    if plot_type == "bar":
        shap.plots.bar(shap_values, show_data=True, show=show)
    else:
        shap.plots.waterfall(shap_values, show=show)
    if save:
        plt.savefig(os.path.join(save_path, f"{save_name}_{plot_type}.{file_type}"), bbox_inches='tight')
        plt.clf()


def plot_distribution(data, name, xlabel="Trust Values", figsize=(6, 4), xticks=None):
    mean, sigma = np.mean(data), np.std(data)
    # create histogram
    fig, ax = plt.subplots(figsize=figsize)
    n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.5)

    # plot normal distribution curve
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mean))**2))
    ax.plot(bins, y, '--')

    # set plot parameters
    shap = scipy.stats.shapiro(data)
    print(name, shap)
    pval = shap.pvalue
    if pval < 0.05:
        if pval < 0.0001:
            title = f'{name}, $W={shap.statistic:.2f}$, $p<{0.0001}$*'
        else:
            title = f'{name}, $W={shap.statistic:.2f}$, $p={pval:.4f}$*'
    else:
        title = f'{name}, $W={shap.statistic:.2f}$, $p={pval:.4f}$'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True)
    if xticks is not None:
        ax.set_xticks(xticks)
    
    return fig
    
    
def plot_boxplot(tables, colors, xticks, yticks, ylabel, xlabel=None, title=None, figsize=(6, 4), regex=None,
                 yticks_labels=None):
    fig, ax = plt.subplots(figsize=figsize)
    linewidth = 1.75

    outline_col = "dimgray"
    for (i, table), col in zip(enumerate(tables), colors):
        if regex is not None:
            table = table.filter(regex=regex)
        box = ax.boxplot(table, 
                         positions=[i], widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor=col, color=outline_col, linewidth=linewidth),
                         medianprops=dict(color=outline_col, linewidth=linewidth),
                         whiskerprops=dict(linewidth=linewidth, color=outline_col), 
                         capprops=dict(linewidth=linewidth, color=outline_col),
                         flierprops=dict(marker='o', markerfacecolor=col, markeredgecolor=outline_col, markeredgewidth=0, 
                                         markersize=7))

    # set the labels and title
    ax.set_xticks(ticks=np.arange(len(xticks)),labels=xticks)
    ax.set_yticks(yticks, labels=yticks_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig

def plot_boxplot_testing(tables, colors, xticks, yticks, ylabel, pval,
                         xlabel=None, title=None, figsize=(6, 4), regex=None,
                         yticks_labels=None):
    fig, ax = plt.subplots(figsize=figsize)
    linewidth = 1.75

    outline_col = "dimgray"
    boxes = []
    for (i, table), col in zip(enumerate(tables), colors):
        if regex is not None:
            table = table.filter(regex=regex)
        box = ax.boxplot(table, 
                         positions=[i], widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor=col, color=outline_col, linewidth=linewidth),
                         medianprops=dict(color=outline_col, linewidth=linewidth),
                         whiskerprops=dict(linewidth=linewidth, color=outline_col), 
                         capprops=dict(linewidth=linewidth, color=outline_col),
                         flierprops=dict(marker='o', markerfacecolor=col, markeredgecolor=outline_col, markeredgewidth=0, 
                                         markersize=7))
        # print(box)
        boxes.append(box)
    
    y = 7.5
    x1 = boxes[0]['caps'][0].get_xdata().mean()
    x2 = boxes[1]['caps'][0].get_xdata().mean()
    ax.plot([x1, x2], [y, y], color=outline_col, linewidth=1.25)
    ax.plot([x1, x1], [y, y-0.2], color=outline_col, linewidth=1.25)
    ax.plot([x2, x2], [y, y-0.2], color=outline_col, linewidth=1.25)
    p_value = f'$p={pval:.4f}$' if pval > 0.0001 else f'$p<{0.0001}$'
    ax.text((x1+x2)/2, y+0.1, p_value, ha='center', fontsize=8)
    
    # set the labels and title
    ax.set_xticks(ticks=np.arange(len(xticks)),labels=xticks)
    ax.set_yticks(yticks, labels=yticks_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig

def plot_barplot(data, colors, labels, categories, title, figsize=(6, 4)):
    linewidth = 1.75

    fig, ax = plt.subplots(figsize=figsize)
    handles = []
    left = np.zeros(len(tables))
    for c, category in enumerate(categories):
        data_per_category = data[:, j, :]
        handles.append(ax.barh(labels, data_per_category[:, c], left=left, color=colors[c], label=category, height=0.5))

        left += data_per_category[:, c]

    # set the labels and title
    ax.set_xlabel('Number of Responses')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(handles=handles[::1], loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    
    return fig