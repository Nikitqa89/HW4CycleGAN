import argparse
import itertools
import datetime
import time
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from models import *
from datasets import *
from utils import *

# растягиваем картинки до нужного размера
def get_transforms(img_height, img_width):
    return [
        transforms.Resize((img_height, img_width), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]


def sample_images(batches_done, val_dataloader, G_AB, G_BA, Tensor, dataset_name):
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = imgs["A"].type(Tensor)
    fake_B = G_AB(real_A)
    real_B = imgs["B"].type(Tensor)
    fake_A = G_BA(real_B)
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, f"images/{dataset_name}/{batches_done}.png", normalize=False)

def train(opt):
    os.makedirs(f"images/{opt.dataset_name}", exist_ok=True)
    os.makedirs(f"saved_models/{opt.dataset_name}", exist_ok=True)

    # Определение функций потерь
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    cuda = torch.cuda.is_available()
    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Инициализация генераторов и дискриминаторов
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    # Перенос моделей и функций потерь на GPU (если CUDA включен)
    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    # Загрузка моделей, если training продолжается с заданной эпохи
    if opt.epoch != 0:
        G_AB.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/G_AB_{opt.epoch}.pth"))
        G_BA.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/G_BA_{opt.epoch}.pth"))
        D_A.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/D_A_{opt.epoch}.pth"))
        D_B.load_state_dict(torch.load(f"saved_models/{opt.dataset_name}/D_B_{opt.epoch}.pth"))

    # Инициализация весов с нормальным распределением
    else:
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Оптимизаторы
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Планировщики learning rate
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Буферы для фейковых изображений (смягчает колебания при обучении дискриминаторов)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Аугментации и трансформации
    transforms_ = get_transforms(opt.img_height, opt.img_width)
    data_path = os.path.join(opt.data_path, opt.dataset_name)

    # Даталоадеры для обучения и валидации
    dataloader = DataLoader(
        ImageDataset(data_path, transforms_=transforms_, unaligned=True),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    val_dataloader = DataLoader(
        ImageDataset(data_path, transforms_=transforms_, unaligned=True, mode="test"),
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )

    prev_time = time.time()

    # Главный цикл по эпохам
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Подготовка входных изображений
            real_A = batch["A"].type(Tensor)
            real_B = batch["B"].type(Tensor)

            # Метки для real и fake
            valid = torch.ones((real_A.size(0), *D_A.output_shape), device=real_A.device)
            fake = torch.zeros((real_A.size(0), *D_A.output_shape), device=real_A.device)

            # Тренировка генераторов
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()

            # Identity loss (сохранять исходный стиль, если не нужно переводить)
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) * 0.5

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) * 0.5

            # Cycle-consistency loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) * 0.5

            # Общий loss для генераторов
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # Тренировка дискриминатора A
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real + loss_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # Тренировка дискриминатора B
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real + loss_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # Общий дискриминаторный loss
            loss_D = (loss_D_A + loss_D_B) * 0.5

            # Логирование и ETA
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            print(
                f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}, adv: {loss_GAN.item():.4f}, "
                f"cycle: {loss_cycle.item():.4f}, identity: {loss_identity.item():.4f}] ETA: {time_left}"
            )

            # Сохранение примеров
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done, val_dataloader, G_AB, G_BA, Tensor, opt.dataset_name)

        # Обновление learning rate
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Сохранение чекпоинтов моделей
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(G_AB.state_dict(), f"saved_models/{opt.dataset_name}/G_AB_{epoch}.pth")
            torch.save(G_BA.state_dict(), f"saved_models/{opt.dataset_name}/G_BA_{epoch}.pth")
            torch.save(D_A.state_dict(), f"saved_models/{opt.dataset_name}/D_A_{epoch}.pth")
            torch.save(D_B.state_dict(), f"saved_models/{opt.dataset_name}/D_B_{epoch}.pth")

    # Сохранение финальных весов
    torch.save(G_AB.state_dict(), f"saved_models/{opt.dataset_name}/G_AB_final.pth")
    torch.save(G_BA.state_dict(), f"saved_models/{opt.dataset_name}/G_BA_final.pth")
    torch.save(D_A.state_dict(), f"saved_models/{opt.dataset_name}/D_A_final.pth")
    torch.save(D_B.state_dict(), f"saved_models/{opt.dataset_name}/D_B_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="стартовая эпоха")
    parser.add_argument("--n_epochs", type=int, default=100, help="общее количество эпох обучения")
    parser.add_argument("--dataset_name", type=str, default="AIvazovsky", help="папка с датасетом")
    parser.add_argument("--data_path", type=str, default="data", help="root path to datasets")
    parser.add_argument("--batch_size", type=int, default=1, help="размер батча")
    parser.add_argument("--lr", type=float, default=0.0002, help="начальная скорость обучения")
    parser.add_argument("--b1", type=float, default=0.5, help="Параметр beta1 для Adam-оптимизатора")
    parser.add_argument("--b2", type=float, default=0.999, help="Параметр beta2 для Adam-оптимизатора")
    parser.add_argument("--decay_epoch", type=int, default=50, help="С какой эпохи начать линейное уменьшение скорости обучения")
    parser.add_argument("--n_cpu", type=int, default=8, help="Количество потоков CPU для загрузки данных")
    parser.add_argument("--img_height", type=int, default=256, help="Высота входного изображения")
    parser.add_argument("--img_width", type=int, default=256, help="Ширина входного изображения")
    parser.add_argument("--channels", type=int, default=3, help="Количество каналов (3 — RGB, 1 — grayscale)")
    parser.add_argument("--sample_interval", type=int, default=500, help="Частота сохранения изображений во время обучения (в итерациях)")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="Частота сохранения модели (в эпохах)")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="Количество резидуальных блоков в генераторе"
                                                                         "(рекомендуется 9 для 256x256+, 6 для меньшего разрешения)")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="Вес cycle-consistency потерь (cyc_loss)")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="Вес identity потерь (id_loss) "
                                                                     "уменьшает искажения при переводе в тот же домен")

    opt = parser.parse_args()

    import multiprocessing
    multiprocessing.freeze_support()
    train(opt)