import os
import math
import dicom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset



def build_dataset(test_patient_number_list, mm, norm_range=(-1024.0, 3072.0)):
    data_path = '/data1/AAPM-Mayo-CT-Challenge/'
    patients_list = [data for data in os.listdir(data_path) if 'zip' not in data]
    assert len(patients_list) == 10

    input_img = []
    target_img = []
    test_input_img = []
    test_target_img = []

    for patient_ind, patient in enumerate(patients_list):
        patient_path = os.path.join(data_path, patient)

        if patient not in test_patient_number_list:
            input_path = [data for data in os.listdir(patient_path) if "quarter" in data and mm in data and "sharp" not in data][0]
            target_path = [data for data in os.listdir(patient_path) if "full" in data and mm in data and "sharp" not in data][0]

            for io in [input_path, target_path]:
                full_pixels = get_pixels_hu(load_scan(os.path.join(patient_path, io) + '/'))
                if io == input_path:
                    for img_ind in range(full_pixels.shape[0]):
                        input_img.append(NORMalize(full_pixels[img_ind], norm_range[0], norm_range[1]))
                else:
                    for img_ind in range(full_pixels.shape[0]):
                        target_img.append(NORMalize(full_pixels[img_ind], norm_range[0], norm_range[1]))
        else:
            test_input_path = [data for data in os.listdir(patient_path) if "quarter" in data and mm in data and "sharp" not in data][0]
            test_target_path  = [data for data in os.listdir(patient_path) if "full" in data and mm in data and "sharp" not in data][0]
            for io in [test_input_path, test_target_path]:
                full_pixels = get_pixels_hu(load_scan(os.path.join(patient_path, io) + '/'))
                if io == test_input_path:
                    for img_ind in range(full_pixels.shape[0]):
                        test_input_img.append(NORMalize(full_pixels[img_ind], norm_range[0], norm_range[1]))
                else:
                    for img_ind in range(full_pixels.shape[0]):
                        test_target_img.append(NORMalize(full_pixels[img_ind], norm_range[0], norm_range[1]))

    return input_img, target_img, test_input_img, test_target_img


def load_scan(path):
    slices = [dicom.read_file(path + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def NORMalize(image, MIN_B, MAX_B):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   #image[image>1] = 1.
   #image[image<0] = 0.
   return image

def DENORMalize(image, MIN_B, MAX_B):
    image = image * (MAX_B - MIN_B) + MIN_B
    return image


class train_dcm_data_loader(Dataset):
    def __init__(self, input_lst, target_lst, crop_size=None, crop_n=None):
        self.input_lst = input_lst
        self.target_lst = target_lst
        self.crop_size = crop_size
        self.crop_n = crop_n

    def __getitem__(self, idx):
        input_img = self.input_lst[idx]
        target_img = self.target_lst[idx]

        if self.crop_n:
            assert input_img.shape == target_img.shape
            crop_input = []
            crop_target = []
            h, w = input_img.shape
            new_h, new_w = self.crop_size, self.crop_size
            for _ in range(self.crop_n):
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
                input_img_ = input_img[top:top + new_h, left:left + new_w]
                target_img_ = target_img[top:top + new_h, left:left + new_w]
                crop_input.append(input_img_)
                crop_target.append(target_img_)
            crop_input = np.array(crop_input)
            crop_target = np.array(crop_target)

            sample = (crop_input, crop_target)
            return sample
        else:
            sample = (input_img, target_img)
            return sample

    def __len__(self):
        return len(self.input_lst)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=4096):
    L = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=4096, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


class Structural_loss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(Structural_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        MS_SSIM = msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return 1 - MS_SSIM


class LambdaLR_():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class Generator_2D_CNN(nn.Module):
    def __init__(self):
        super(Generator_2D_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv_ = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, img):
        out = self.relu(self.conv_first(img))
        for _ in range(6):
            out = self.relu(self.conv_(out))
        out = self.relu(self.conv_last(out))
        return out


class Discriminator_2D(nn.Module):
    def __init__(self, input_size=64):
        super(Discriminator_2D, self).__init__()

        def after_conv_size_c(input_size, kernel_size_list, stride_list):
            cal = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for i in range(1, len(kernel_size_list)):
                cal = (cal - kernel_size_list[i]) // stride_list[i] + 1
            return cal

        def discriminator_block(in_filters, out_filters, stride):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, padding=0)]
            layers.append(nn.LeakyReLU(0.2))
            return layers

        layers = []
        for in_filters, out_filters, stride in [(1,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride))

        self.fc_size = after_conv_size_c(input_size, [3,3,3,3,3,3], [1,2,1,2,1,2])
        self.cnn = nn.Sequential(*layers)
        self.leaky = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(256* self.fc_size* self.fc_size, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, img):
        out = self.cnn(img)
        out = out.view(-1, 256* self.fc_size* self.fc_size)
        out = self.fc1(out)
        out = self.leaky(out)
        out = self.fc2(out)
        return out



def calc_gradeint_penalty(discriminator, real_data, fake_data):
    #alpha = torch.rand(real_data.size()[0], 1)
    #alpha = alpha.expand(real_data.size())
    alpha = torch.Tensor(np.random.random((real_data.size(0),1,1,1)))
    alpha = alpha.cuda() if torch.cuda.is_available() else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

