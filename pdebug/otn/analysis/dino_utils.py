import os

from pdebug.otn import manager as otn_manager
from pdebug.utils.env import TORCH_INSTALLED, TORCHVISION_INSTALLED

import matplotlib.pyplot as plt
import typer
from PIL import Image

try:
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    PCA = None

if TORCH_INSTALLED:
    import torch
if TORCHVISION_INSTALLED:
    from torchvision import transforms


@otn_manager.NODE.register(name="imgdir_to_dino_vis")
def imgdir_to_dino_vis(
    path: str,
    output: str = "tmp_dino_vis",
    cache: bool = False,
    log_pyplot: bool = True,
):
    """Do dino_v2 feature visualization."""
    if cache and os.path.exists(output):
        typer.echo(typer.style(f"Found {output}, skip", fg=typer.colors.WHITE))
        return output
    os.makedirs(output, exist_ok=True)

    device = torch.device("cuda:0")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    transform1 = transforms.Compose(
        [
            transforms.Resize(520),
            transforms.CenterCrop(
                518
            ),  # should be multiple of model patch_size
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.2),
        ]
    )

    patch_size = model.patch_size  # patchsize=14

    # 520//14
    patch_h = 520 // patch_size
    patch_w = 520 // patch_size

    # feat_dim = 384 # vits14
    # feat_dim = 768 # vitb14
    feat_dim = 1024  # vitl14
    # feat_dim = 1536 # vitg14

    folder_path = path
    total_features = []
    with torch.no_grad():
        for img_path in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_path)
            img = Image.open(img_path).convert("RGB")
            img_t = transform1(img)

            features_dict = model.forward_features(
                img_t.unsqueeze(0).to(device)
            )
            features = features_dict["x_norm_patchtokens"].cpu()
            total_features.append(features)

    num_images = len(total_features)

    total_features = torch.cat(total_features, dim=0)

    # First PCA to Seperate Background
    # sklearn expects 2d array for training
    total_features = total_features.reshape(-1, feat_dim)  # 4(*H*w, 1024)

    assert PCA, "sklearn is required."
    pca = PCA(n_components=3)
    pca.fit(total_features)
    pca_features = pca.transform(total_features)

    if log_pyplot:
        # visualize PCA components for finding a proper threshold
        # 3 histograms for 3 components
        plt.clf()
        plt.title("pca_features_hist")
        plt.subplot(2, 2, 1)
        plt.hist(pca_features[:, 0])
        plt.subplot(2, 2, 2)
        plt.hist(pca_features[:, 1])
        plt.subplot(2, 2, 3)
        plt.hist(pca_features[:, 2])
        # plt.show()
        # plt.close()
        savename = os.path.join(output, "1.pca_features_hist.png")
        plt.savefig(savename)

    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (
        pca_features[:, 0].max() - pca_features[:, 0].min()
    )

    if log_pyplot:
        plt.clf()
        plt.title("pca_features_0")
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(
                pca_features[
                    i * patch_h * patch_w : (i + 1) * patch_h * patch_w, 0
                ].reshape(patch_h, patch_w)
            )
        # plt.show()
        savename = os.path.join(output, "2.pca_features_0.png")
        plt.savefig(savename)

    # segment/seperate the backgound and foreground using the first component
    bg_threshold = 0.35  # from first histogram
    pca_features_bg = pca_features[:, 0] > bg_threshold
    pca_features_fg = ~pca_features_bg

    if log_pyplot:
        # plot the pca_features_bg
        plt.clf()
        plt.title("pca_features_bg")
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(
                pca_features_bg[
                    i * patch_h * patch_w : (i + 1) * patch_h * patch_w
                ].reshape(patch_h, patch_w)
            )
        # plt.show()
        savename = os.path.join(output, "3.pca_features_bg.png")
        plt.savefig(savename)

    # 2nd PCA for only foreground patches
    pca.fit(total_features[pca_features_fg])
    pca_features_left = pca.transform(total_features[pca_features_fg])

    for i in range(3):
        # min_max scaling
        pca_features_left[:, i] = (
            pca_features_left[:, i] - pca_features_left[:, i].min()
        ) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())

    pca_features_rgb = pca_features.copy()
    # for black background
    pca_features_rgb[pca_features_bg] = 0
    # new scaled foreground features
    pca_features_rgb[pca_features_fg] = pca_features_left

    # reshaping to numpy image format
    pca_features_rgb = pca_features_rgb.reshape(
        num_images, patch_h, patch_w, 3
    )
    if log_pyplot:
        plt.clf()
        plt.title("pca_features_rgb(fg)")
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(pca_features_rgb[i])
        # plt.show()
        savename = os.path.join(output, "4.pca_features_rgb(fg).png")
        plt.savefig(savename)

        plt.clf()
        plt.title("images")
        for i, img_path in enumerate(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_path)
            img = Image.open(img_path).convert("RGB").resize((1000, 700))
            plt.subplot(2, 2, i + 1)
            plt.imshow(img)
            if i >= 4:
                break
        # plt.show()
        savename = os.path.join(output, "5.images.png")
        plt.savefig(savename)

    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return output


if __name__ == "__main__":
    typer.run(imgdir_to_dino_vis)
