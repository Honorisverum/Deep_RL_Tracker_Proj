import loader
import matplotlib.pyplot as plt
import network
import torch
import functions
import numpy as np

"""
=================================================
        PARAMETRS, CONSTANTS, NETWORK
=================================================
"""

image_size = (32, 32)
# ('regular', c) : constant c
# ('exponent', c, b) : exponentially decaying with epochs c * exp(-b * epoch)
# ('polynomially', c, k) : inverse polynomial decaying with epochs c / epoch ** k
SIGMA_TYPE = ('exponent', 0.05, 0.1)
N = 50
T = 20
initial_stage = 50
late_stage = 30
LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)
EPS = 1e-08
MODEL_DIM = 200
epochs = initial_stage + late_stage
rewards_curve = []
weights_curve = []
weights_grad_curve = []

# save and load
SAVE_EVERY = 10  # None if no need to save
save_path = 'weights/' + 'weights.pt'
IS_LOAD = True  # True if need to load, False if not
load_path = 'weights/' + 'weights.pt'

# load training/test set
training_set_titles = ['Vid_A_ball',
                       'Vid_B_cup',
                       'Vid_D_person',
                       'Vid_H_panda',
                       'trans',
                       'shaking']
                       #'car11',
                       #'trellis',
                       #'singer1',
                       #'skating1']
training_set_videos = loader.load_videos(training_set_titles, T, image_size)

test_set_titles = ['gym',
                   'Vid_G_rubikscube',
                   'singer2',
                   'Vid_C_juice']
test_set_videos = loader.load_videos(test_set_titles, T, image_size)

# create net
if IS_LOAD:
    net = torch.load(load_path)
else:
    net = network.CNN_LSTM(MODEL_DIM)

# torch.save(model, 'filename.pt')
# model = torch.load('filename.pt')

# choose optimizer
optimizer = torch.optim.Adam(net.parameters(),
                             lr=LEARNING_RATE,
                             betas=BETAS,
                             eps=EPS)


"""
=================================================
                TRAINING PHASE 
=================================================
"""


for epoch in range(1, epochs + 1):

    # reward func depend on stage
    if epoch <= initial_stage:
        reward_func = functions.reward1
    else:
        reward_func = functions.reward2

    # calculate sigma for this epoch
    sigma = functions.compute_sigma(SIGMA_TYPE, epoch)

    # weight info for episode
    ep_weights = []
    ep_weights_grad = []
    ep_reward = []

    for video in training_set_videos:

        # accumulate reward/weights for info
        ep_video_reward = 0
        ep_video_weights = 0
        ep_video_weights_grad = 0

        for gt, (images, _) in zip(video.ground_truth_loader, video.frames_loader):

            # compute location vec
            s_t = functions.gt_location_vec(gt)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass to get output
            outputs = net(images, s_t)

            # sample predictions
            predictions = functions.sample_predictions(outputs, sigma, N).detach()

            # calculate rewards
            rewards = functions.compute_rewards(predictions, gt, reward_func, N)

            # calculate baseline
            baseline = functions.compute_baseline(rewards)
            #baseline = functions.compute_log_baseline(rewards, predictions)

            # calculate diff and loss
            differences = functions.calculate_diff(outputs, predictions, sigma).detach()
            loss = functions.compute_loss(rewards, baseline, outputs, differences)

            # Getting gradients
            loss.backward()

            # Updating parameters
            optimizer.step()

            if images.size(0) == T:
                ep_video_reward += baseline
                ep_video_weights += functions.compute_weights(net)
                ep_video_weights_grad += functions.compute_weights_grad(net)

        # print info for ep for this video
        iteration_info_format = {
            'video_title': video.title,
            'epoch_num': epoch,
            'mean_reward': round(ep_video_reward / (video.complete_sequences * T), 3)
        }
        print("Training end for {video_title} |"
              " Epoch: {epoch_num} |"
              " Mean Reward Per Frame: {mean_reward}".format(**iteration_info_format))

        # for curves
        ep_weights.append(ep_video_weights / video.complete_sequences)
        ep_weights_grad.append(ep_video_weights_grad / video.complete_sequences)
        ep_reward.append(ep_video_reward / (video.complete_sequences * T))

    if SAVE_EVERY is not None:
        if epoch % SAVE_EVERY:
            torch.save(net, save_path)

    weights_curve.append(np.mean(ep_weights))
    weights_grad_curve.append(np.mean(ep_weights_grad))
    rewards_curve.append(np.mean(ep_reward))

    # for divide info on ep blocks
    print("")

# final save
if SAVE_EVERY is not None:
    torch.save(net, save_path)


"""
=================================================
                TEST PHASE 
=================================================
"""


for video in test_set_videos:

    start_from_ind = 1
    left_frames = video.set_len

    while left_frames > 0:

        img_seq = video.sample_frames(start_from_ind, left_frames)
        gt_seq = video.sample_gt(start_from_ind, left_frames)

        s_t = functions.gt_location_vec(gt_seq)

        outputs = net(img_seq, s_t)

        overlaps = functions.compute_overlaps(outputs, gt_seq)

        if 0 in overlaps:
            ind_0 = overlaps.index(0) + 1
            video.test_fails += 1
            video.test_predictions.extend(overlaps[:ind_0])

            if left_frames <= ind_0 + 4:
                video.test_predictions.extend([0]*left_frames)
                left_frames = 0
            else:
                video.test_predictions.extend([0, 0, 0, 0])
                start_from_ind = ind_0 + 4 + 1
                left_frames -= ind_0 + 4
        else:
            video.test_predictions.extend(overlaps)
            left_frames = 0


"""
=================================================
                QUALITY CONTROL 
=================================================
"""


print("RESULTS:")


for video in test_set_videos:
    overlaps = video.test_predictions
    non_zer = np.sum([1 for x in overlaps if x > 0])
    mean_acc = np.sum(overlaps) / non_zer
    video_info_format = {
        'video_title': video.title,
        'frames_num': video.set_len,
        'accuracy': mean_acc,
        'fails': video.test_fails
    }
    print("Test end for {video_title} with"
          "{frames_num} frames |"
          " Accuracy: {accuracy} |"
          " Number of fails: {fails}".format(**video))


# learning curve
plt.plot(rewards_curve, color='orange')
plt.xlabel('iteration each T frames')
plt.ylabel('cumulative reward')
plt.savefig('results/learning_curve.png')
plt.close()

# weights curve
plt.plot(weights_curve, color='red')
plt.xlabel('iteration each T frames')
plt.ylabel('weights modulus')
plt.savefig('results/weights_curve.png')
plt.close()

# weights grad curve
plt.plot(weights_grad_curve, color='blue')
plt.xlabel('iteration each T frames')
plt.ylabel('weights grad modulus')
plt.savefig('results/weights_grad_curve.png')
plt.close()


