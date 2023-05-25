import torch
from captum.attr import GradientShap, DeepLift, IntegratedGradients, LayerGradCam, LayerAttribution, NoiseTunnel, GuidedBackprop, GuidedGradCam, Saliency
import numpy as np
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------
# Explanation methods
# -----------------------------------------------------------------

# attribution functions
def attribute_image_features(model, algorithm, input, target, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input, target=target, **kwargs)
    return tensor_attributions

def IG(model, sample, target, modifier="base"):
    ig = IntegratedGradients(model)
    # baseline = torch.zeros(1,3,32,32).cuda()
    # attribution = ig.attribute(sample.reshape(1,3,32,32).cuda(), baseline, target=target, return_convergence_delta=False)
    attr_ig, delta = attribute_image_features(model, ig, sample, target, baselines=sample * 0,
                                              return_convergence_delta=True, internal_batch_size=32, n_steps=25)
    # attribution = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    attribution = attr_ig.detach().cpu()
    return attribution

def IG_SG(model, sample, target, modifier):
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)#.to(dtype=torch.half)#.half()
    attr_ig_nt = attribute_image_features(model, nt, sample, target, baselines=sample * 0, nt_type=modifier, #variance, square
                                      nt_samples=10, stdevs=0.2, internal_batch_size=32, n_steps=25)
    # attribution = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attribution = attr_ig_nt.detach().cpu()
    return attribution

def GB(model, sample, target, modifier="base"):
    gb = GuidedBackprop(model)
    # baseline = torch.zeros(1,3,32,32).cuda()
    # attribution = ig.attribute(sample.reshape(1,3,32,32).cuda(), baseline, target=target, return_convergence_delta=False)
    attr_ig = attribute_image_features(model, gb, sample, target)
    # attribution = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    attribution = attr_ig.detach().cpu()
    return attribution

def GB_SG(model, sample, target, modifier):
    gb = GuidedBackprop(model)
    nt = NoiseTunnel(gb)
    # baseline = torch.zeros(1,3,32,32).cuda()
    # attribution = ig.attribute(sample.reshape(1,3,32,32).cuda(), baseline, target=target, return_convergence_delta=False)
    # attr_ig_nt = attribute_image_features(model, nt, sample, target, nt_samples=100, nt_type='vargrad')
    attr_ig_nt = attribute_image_features(model, nt, sample, target, nt_type=modifier,
                                          nt_samples=100, nt_samples_batch_size=8)
    # attr_ig_nt = attribute_image_features(model, nt, sample, target, nt_type=modifier,
    #                                       nt_samples=10, internal_batch_size=32, n_steps=25)
    # attribution = np.transpose(attr_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
    attribution = attr_ig_nt.detach().cpu()
    return attribution

def GradShap(model, sample, target):
    gradient_shap = GradientShap(model)
    # baseline = torch.randn(1, 3, 32, 32).cuda()
    attri_gs = attribute_image_features(model, gradient_shap, sample, target, baselines=sample*0, n_samples=1, stdevs=0.0)
    attribution = np.transpose(attri_gs.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return attribution

def Deeplift(model, sample, target):
    deeplift = DeepLift(model)
    attribution = deeplift.attribute(sample.reshape(1,3,32,32).cuda(), target=target)
    return attribution.detach().squeeze().cpu() #.numpy()

def GradCAM(model, sample, target, modifier="base", size=224):
    sample = Variable(sample).cuda()
    layer_gc = LayerGradCam(model, model.layer4)
    attr = layer_gc.attribute(sample, target=target)
    m = torch.nn.Upsample(size=(size, size), mode="bilinear", align_corners=True)
    upsampled_attr = m(attr) #
    # upsampled_attr = LayerAttribution.interpolate(attr, (224, 224),
    #                                               interpolate_mode="bilinear")
    return upsampled_attr.detach().cpu() #.numpy()

def GradShap_SG(model, sample, target):
    gradient_shap = GradientShap(model)
    nt = NoiseTunnel(gradient_shap)
    # baseline = torch.zeros(1,3,32,32).cuda()
    # attribution = ig.attribute(sample.reshape(1,3,32,32).cuda(), baseline, target=target, return_convergence_delta=False)
    attr_ig_nt = attribute_image_features(model, nt, sample, target, baselines=sample * 0,  nt_samples=100, nt_type='vargrad', stdevs=0.2)
    attribution = np.transpose(attr_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return attribution

def SaliencyMap(model, sample, target, modifier="base"):
    sample = Variable(sample).cuda()
    saliency = Saliency(model)
    # Computes saliency maps for class 3.
    attribution = saliency.attribute(sample, target=target)
    return attribution.detach().cpu()

# ------------------------------------------------------------------
# Explanation Manipulation Methods by Heo et al.
# https://github.com/rmrisforbidden/Fooling_Neural_Network-Interpretations
# ------------------------------------------------------------------

def normalize(attribution):
    attribution = attribution - attribution.min(dim=1)[0].min(dim=1)[0].min(dim=1)[0].reshape(-1, 1, 1, 1)
    attribution = attribution / (attribution.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].reshape(-1, 1, 1, 1) + 1e-8)
    return attribution

def location(attribution, size, device, keep_batch=False):
    """
    :param attribution: tensor (batch, channel, height, width)
    :return: loss
    """
    # attribution = normalize(attribution)
    l, n = attribution.shape[2], attribution.shape[0]
    mask = torch.ones_like(attribution)
    k = (l // size)
    mask[:, :, k:-k, k:-k] = torch.zeros((l - 2 * k, l - 2 * k)).to(device)

    attribution = attribution * mask
    return attribution

def corner(attribution, size, device, keep_batch=False):
    """
    :param attribution: tensor (batch, channel, height, width)
    :return: loss
    """
    attribution = normalize(attribution)

    mask = torch.zeros_like(attribution)
    h, w = attribution.shape[-2], attribution.shape[-1] # height, width
    k_w = (w // size) * 2
    k_h = (h // size) * 2
    mask[:, :, :k_h, :k_w] = (torch.ones((k_h, k_w), dtype=torch.float32)).to(device)

    return attribution * mask


def center_mass(attribution, attribution_ori, device, keep_batch=False):
    b, c, h, w = attribution.shape
    attribution = normalize(attribution).to(device)
    attribution_ori = normalize(attribution_ori).to(device)

    attribution_sum = attribution.sum(dim=(1, 2, 3))
    attribution_ori_sum = attribution_ori.sum(dim=(1, 2, 3))

    h_vector = torch.arange(1, h+1).reshape(1, 1, h, 1).to(device)
    w_vector = torch.arange(1, w+1).reshape(1, 1, 1, w).to(device)

    attribution_h = torch.abs(attribution.sum(dim=3, keepdim=True) * h_vector).sum(dim=2).sum(dim=1).squeeze() \
                    / (attribution_sum + 1e-8)
    attribution_w = torch.abs(attribution.sum(dim=2, keepdim=True) * w_vector).sum(dim=3).sum(dim=1).squeeze() \
                    / (attribution_sum + 1e-8)
    attribution_ori_h = torch.abs(attribution_ori.sum(dim=3, keepdim=True) * h_vector).sum(dim=2).sum(dim=1).squeeze() \
                        / (attribution_ori_sum + 1e-8)
    attribution_ori_w = torch.abs(attribution_ori.sum(dim=2, keepdim=True) * w_vector).sum(dim=3).sum(dim=1).squeeze() \
                        / (attribution_ori_sum + 1e-8)

    if keep_batch:
        distance = (attribution_h - attribution_ori_h)**2 + (attribution_w - attribution_ori_w)**2
        distance = torch.sqrt(distance/2) / (w - 1)
        return distance

    distance = -1 * ((attribution_h - attribution_ori_h).abs() + (attribution_w - attribution_ori_w).abs()).mean()
    return distance