from torch import nn

def shared_mlp(
    channels: list[int],
    bn: bool = True,
    last_bn: bool = True,
    last_relu: bool = True,
) -> nn.Sequential:
    """
    Shared MLP = point-wise 1x1 Conv over points.

    channels: npr. [3, 64, 128, 1024]
    bn:       koristi BatchNorm iza Conv1d (default True)
    last_bn:  koristi BN i na zadnjem sloju (default True)
    last_relu:koristi ReLU i na zadnjem sloju (default True)

    Default ponašanje je BN+ReLU nakon svakog sloja.
    """
    layers = []
    n_layers = len(channels) - 1

    for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
        is_last = (i == n_layers - 1)

        layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=not bn))

        if bn and (last_bn or not is_last):
            layers.append(nn.BatchNorm1d(out_ch))

        if last_relu or not is_last:
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
