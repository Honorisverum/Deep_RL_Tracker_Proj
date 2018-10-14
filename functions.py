import math
import torch

"""
=================================================
    SAMPLE PREDICTIONS AND CALCULATE REWARDS
=================================================
"""


# sample predictions
# network_output : (T, 4)
# return size : (T, N, 4)
def sample_predictions(network_output, sig, N):
    tens_list = []
    for i in range(network_output.size(0)):
        dist = torch.distributions.multivariate_normal.MultivariateNormal(network_output[i], (sig ** 2) * torch.eye(4))
        sample = dist.rsample(sample_shape=torch.Size([N])).unsqueeze(0)
        tens_list.append(sample)
    return torch.cat(tens_list, 0)


# first type of reward
def reward1(pred, gt):
    subtraction = abs(pred - gt)
    return - subtraction.mean().item() / 2 - max(subtraction).item() / 2


# second type of reward
def reward2(pred, gt):
    # calculating length and width of intersect area
    dx = min(pred[0].item() + pred[2].item() / 2, gt[0].item() + gt[2].item() / 2) - \
         max(pred[0].item() - pred[2].item() / 2, gt[0].item() - gt[2].item() / 2)
    dy = min(pred[1].item() + pred[3].item() / 2, gt[1].item() + gt[3].item() / 2) - \
         max(pred[1].item() - pred[3].item() / 2, gt[1].item() - gt[3].item() / 2)

    # intersect square
    if (dx >= 0) and (dy >= 0):
        intersect = dx * dy
    else:
        return 0

    # union area
    union = pred[2].item() * pred[3].item() + gt[2].item() * gt[3].item() - intersect

    return intersect / union


# predictions : (T, N, 4)
# ground truth : (T, 4)
# out : (T, N)
def compute_rewards(predictions, ground_truth, reward_func, N):
    out_rewards = torch.zeros(ground_truth.size(0), N)
    for i in range(ground_truth.size(0)):
        for j in range(N):
            out_rewards[i][j] = reward_func(predictions[i][j], ground_truth[i])
    return out_rewards


"""
=================================================
        BASELINES, LOSS AND SIGMA
=================================================
"""


# create location vector s_t with gt at 1st frame and 0's at all other
def gt_location_vec(gt):
    len = gt.size(0)  # gt sequence size
    return torch.cat((gt[0].unsqueeze(0),
                      torch.zeros((len - 1, 4), dtype=torch.float32)), 0)


# new baseline
# rew : (T, N)
def compute_baseline(rew):
    rew = torch.sum(rew, dim=0)
    return torch.mean(rew, dim=0).item()


# diff : (T, N, 4)
# out : (T, 4)
# rew : (T, N)
def compute_loss(rew, bs, out, diff):
    rew = torch.sum(rew, dim=0) - bs
    rew = rew.unsqueeze(0).unsqueeze(2)
    rew = rew.expand_as(diff)
    out = out.unsqueeze(1).expand_as(diff)
    return torch.sum(diff * out * rew)


def calculate_diff(out, pred, sig):
    out = out.unsqueeze(1).expand_as(pred)
    df = (out - pred) / (sig ** 2)
    return df


# weights for weight change curve
def compute_weights(net):
    ans = 0
    for param in net.parameters():
        ans += math.sqrt(torch.sum(param ** 2).item())
    return ans


# weights grad for weight grad change curve
def compute_weights_grad(net):
    ans = 0
    for param in net.parameters():
        ans += math.sqrt(torch.sum(param.grad ** 2).item())
    return ans


# ('regular', c) : constant c
# ('exponent', c, b) : exponentially decaying with epochs c * exp(-b * epoch)
# ('polynomially', c, k) : inverse polynomial decaying with epochs c / epoch ** k
def compute_sigma(sigma_type, epoch):
    if sigma_type[0] == 'regular':
        return sigma_type[1]
    elif sigma_type[0] == 'exponent':
        return sigma_type[1] * math.exp(- sigma_type[2] * epoch)
    elif sigma_type[0] == 'polynomially':
        return sigma_type[1] / (epoch ** sigma_type[2])


def compute_overlaps(out, gt):
    ans = []
    for out_row, gt_row in zip(out, gt):
        ans.append(compute_overlap(out_row, gt_row))
    return ans


def compute_overlap(out_r, gt_r):

    # calculating length and width of overlap area
    dx = min(out_r[0].item() + out_r[2].item() / 2, gt_r[0].item() + gt_r[2].item() / 2) - \
         max(out_r[0].item() - out_r[2].item() / 2, gt_r[0].item() - gt_r[2].item() / 2)
    dy = min(out_r[1].item() + out_r[3].item() / 2, gt_r[1].item() + gt_r[3].item() / 2) - \
         max(out_r[1].item() - out_r[3].item() / 2, gt_r[1].item() - gt_r[3].item() / 2)

    # overlap square
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0