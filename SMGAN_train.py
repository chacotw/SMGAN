import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from SMGAN_util import train_dcm_data_loader, Generator_2D_CNN, Discriminator_2D, build_dataset, LambdaLR_, weights_init_normal, calc_gradeint_penalty, Structural_loss

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    START_EPOCH = 0
    DECAY_EPOCH = 50
    NUM_EPOCH = 100
    BATCH_SIZE = 1
    CROP_NUMBER = 60  # The number of patches to extract from a single image. --> total batch img is BATCH_SIZE * CROP_NUMBER
    CROP_SIZE = 80
    N_CPU = 30
    CRITIC_ITER = 5
    LEARNING_RATE = 1e-3
    _BETA = 10e-3
    _TAU = 0.89
    _LAMBDA = 10

    save_path = '/home/shsy0404/result/SMGAN_result'

    generator = Generator_2D_CNN()
    discriminator = Discriminator_2D(input_size=CROP_SIZE)

    if torch.cuda.device_count() > 1:
        print("Use {} GPUs".format(torch.cuda.device_count()), "=" * 9)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    generator.to(device)
    discriminator.to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # loss
    criterion_GAN = nn.MSELoss()
    criterion_Structure = Structural_loss()
    criterion_Sensitive = nn.L1Loss()

    # optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR_(NUM_EPOCH, START_EPOCH, DECAY_EPOCH).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR_(NUM_EPOCH, START_EPOCH, DECAY_EPOCH).step)

   # input & target data
    input_dir, target_dir, test_input_dir, test_target_dir = build_dataset(['L067', 'L291'], "3mm", norm_range=(-1024.0, 3072.0))
    train_dcm = train_dcm_data_loader(input_dir, target_dir, crop_size=CROP_SIZE, crop_n=CROP_NUMBER)
    train_loader = DataLoader(train_dcm, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CPU, drop_last=True)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    target_real = Variable(Tensor(BATCH_SIZE*CROP_NUMBER).fill_(1.0), requires_grad=False)
    target_real = target_real.reshape(-1,1)
    target_fake = Variable(Tensor(BATCH_SIZE*CROP_NUMBER).fill_(0.0), requires_grad=False)
    target_fake = target_fake.reshape(-1,1)

    for epoch in range(START_EPOCH, NUM_EPOCH):
        start_time = time.time()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.reshape(-1, CROP_SIZE, CROP_SIZE).to(device)
            targets = targets.reshape(-1, CROP_SIZE, CROP_SIZE).to(device)

            input_img = torch.tensor(inputs, requires_grad=True).unsqueeze(1).to(device)
            target_img = torch.tensor(targets).unsqueeze(1).to(device)
            if torch.cuda.is_available():
                input_img = input_img.type(torch.cuda.FloatTensor)
                target_img = target_img.type(torch.cuda.FloatTensor)

            ######## Train D ########
            optimizer_D.zero_grad()

            fake_img = generator(input_img)
            real_valid = discriminator(input_img)
            loss_d_real = criterion_GAN(real_valid, target_real)
            fake_valid = discriminator(fake_img)
            loss_d_fake = criterion_GAN(fake_valid, target_fake)

            gradient_penalty = calc_gradeint_penalty(discriminator, input_img.data, fake_img.data)

            loss_D = -loss_d_real + loss_d_fake + _LAMBDA * gradient_penalty
            loss_D.backward()
            optimizer_D.step()

            ######## Train G ########
            optimizer_G.zero_grad()

            if i % CRITIC_ITER == 0:
                fake_img = generator(input_img)
                fake_valid = discriminator(fake_img)
                loss_g = criterion_GAN(fake_valid, target_real)

                # Structure-sensitive loss
                loss_str = criterion_Structure(input_img, fake_img)
                loss_sen = criterion_Sensitive(input_img.data, fake_img.data)
                loss_ssl = _TAU * loss_str + (1-_TAU) * loss_sen

                loss_G = _BETA * loss_g + loss_ssl
                loss_G.backward()
                optimizer_G.step()


            if i % 10 == 0:
                print("EPOCH [{}/{}], STEP [{}/{}]".format(epoch+1, NUM_EPOCH, i+1, len(train_loader)))
                print("Total Loss G: {}, \nLoss Structure: {}, \nLoss Sensitive: {}, \nLoss G: {}, \nTotal Loss D: {} \nLoss D real: {} \nLoss D fake: {} \n==== \n".format(loss_G, loss_str, loss_sen, loss_g, loss_D, loss_d_real, loss_d_fake))

