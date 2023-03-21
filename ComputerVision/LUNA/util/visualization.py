import numpy as np
import matplotlib.pyplot as plt

from dataset import LunaDataset, CtScan

# plt color limits
clim = (-1000.0, 300)

def get_positive_samples(limit=100):
    dataset = LunaDataset()

    positive_samples = []
    # get only the positive samples, i.e. is nodule
    for sample in dataset.nodule_candidate_info:
        if sample.is_nodule:
            positive_samples.append(sample)

        # only collect up to limit samples
        if len(positive_samples) >= limit:
            break

    return positive_samples

def show_nodule_candidate(uid, batch_idx=None):
    # create dataset usign the uid
    dataset = LunaDataset(series_uid=uid)

    # get the positive samples' indices
    pos_idx = [i for i, x in enumerate(dataset.nodule_candidate_info) if x.is_nodule]

    if batch_idx is None:
        if pos_idx:
            batch_idx = pos_idx[0]  # show the first positive sample
        else:
            print('No positive samples, show the negative sample')
            batch_idx = 0

    # get the CT scan of this uid
    ct = CtScan(uid)

    # get the possible nodule region
    ct_chunk, nodule_label, series_uid, center_irc = dataset[batch_idx]
    ct_chunk = ct_chunk[0].numpy()  # convert from tensor to numpy array


    # plotting code from https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p2ch10/vis.py
    fig = plt.figure(figsize=(30, 50))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.ct[int(center_irc[0])], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.ct[:,int(center_irc[1])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.ct[:,:,int(center_irc[2])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_chunk[ct_chunk.shape[0]//2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_chunk[:,ct_chunk.shape[1]//2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_chunk[:,:,ct_chunk.shape[2]//2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index), fontsize=30)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(20)
            plt.imshow(ct_chunk[index], clim=clim, cmap='gray')

    print(series_uid, batch_idx, bool(nodule_label[1]), pos_idx)
