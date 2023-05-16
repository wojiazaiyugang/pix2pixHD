import re
from pathlib import Path

import numpy as np
from typer import Typer

from matplotlib import pyplot as plt

app = Typer()

@app.command()
def main(loss_file: Path,
         save_file: Path):
    lines = loss_file.read_text().splitlines()
    epoch, iters, time, G_GAN, G_GAN_Feat, G_VGG, D_real, D_fake = [], [], [], [], [], [], [], []
    count = 0
    for line in lines:
        # 使用正则表达式匹配epcoh和loss
        match = re.search(
            r"\(epoch: (\d+), iters: (\d+), time: (\d+\.\d+)\) G_GAN: (\d+\.\d+) G_GAN_Feat: (\d+\.\d+) G_VGG: (\d+\.\d+) D_real: (\d+\.\d+) D_fake: (\d+\.\d+) ",
            line)
        if not match:
            continue
        count += 1
        # 避免数据太多影响观察，隔几个取一个
        if count % 500 != 0:
            continue
        items = match.groups()
        epoch.append(float(items[0]))
        iters.append(float(items[1]))
        time.append(float(items[2]))
        G_GAN.append(float(items[3]))
        G_GAN_Feat.append(float(items[4]))
        G_VGG.append(float(items[5]))
        D_real.append(float(items[6]))
        D_fake.append(float(items[7]))

    ax = np.linspace(0, len(epoch), len(epoch))
    # 设置plt 长宽比10:6
    plt.figure(figsize=(10, 3))
    plt.plot(ax, G_GAN, label="G_GAN")
    plt.plot(ax, G_GAN_Feat, label="G_GAN_Feat")
    plt.plot(ax, G_VGG, label="G_VGG")
    plt.plot(ax, D_real, label="D_real")
    plt.plot(ax, D_fake, label="D_fake")
    # 设置y轴范围
    plt.ylim((-0.5, 4))
    # 设置x轴刻度
    # plt.xticks(np.arange(0, len(epoch), step=500))
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(str(save_file))


if __name__ == '__main__':
    app()
