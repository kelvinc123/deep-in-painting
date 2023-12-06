"""Invertible and Deterministic transformation of 4x96x96 <-> 3x512x512"""

import torch

def transform_to_3_512_512(tensor):
    padded_tensor = torch.nn.functional.pad(tensor[:3], (0, 416, 0, 416), mode='constant', value=0)
    # Distributing the fourth channel across the three channels
    for i in range(3):
        padded_tensor[i, :96, 480:512] = tensor[3, :, 32 * i:32 * (i + 1)]

    return padded_tensor

def revert_back_to_4_96_96(tensor):
    # Reconstructing the fourth channel from the three channels
    reconstructed_channel = torch.zeros((1, 96, 96))
    for i in range(3):
        reconstructed_channel[0, :, 32 * i:32 * (i + 1)] = tensor[i, :96, 480:512]

    # Removing padding to get back to 4 x 96 x 96
    reverted_tensor = torch.cat((tensor[:3, :96, :96], reconstructed_channel), dim=0)

    return reverted_tensor