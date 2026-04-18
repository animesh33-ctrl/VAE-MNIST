import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os

# ─── CONFIG ─────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 256
EPOCHS       = 200
LR           = 1e-3
LATENT_DIM   = 32        # bumped from 20
NUM_CLASSES  = 10
SAVE_DIR     = "cvae_outputs"
CKPT_DIR     = "cvae_checkpoints"
CKPT_FILE    = os.path.join(CKPT_DIR, "latest.pt")
BEST_FILE    = os.path.join(CKPT_DIR, "best.pt")
KL_ANNEAL    = True
SAVE_EVERY   = 10        # save checkpoint every N epochs


# ─── ENCODER ────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            flat_size = self.conv(torch.zeros(1, 1, 28, 28)).shape[1]
        self.fc_mu     = nn.Linear(flat_size + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(flat_size + num_classes, latent_dim)

    def forward(self, x, y):
        h = torch.cat([self.conv(x), y], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


# ─── DECODER (bigger) ───────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(latent_dim + num_classes, 256 * 7 * 7)  # 128→256
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        h = self.fc(torch.cat([z, y], dim=1))
        return self.deconv(h.view(z.size(0), 256, 7, 7))


# ─── CVAE ───────────────────────────────────────────────────────
class CVAE(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, log_var):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, x, y):
        mu, log_var = self.encoder(x, y)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z, y), mu, log_var

    def generate(self, z, y):
        return self.decoder(z, y)


# ─── LOSS ───────────────────────────────────────────────────────
def vae_loss(x_recon, x, mu, log_var, kl_weight=1.0):
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum")
    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kl_weight * kl


# ─── ACCURACY ───────────────────────────────────────────────────
def pixel_accuracy(x_recon, x):
    pred = (x_recon > 0.5).float()
    return (pred == x).float().mean().item() * 100.0


# ─── CHECKPOINT ─────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, scheduler, best_loss, path):
    torch.save({
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "best_loss":    best_loss,
        "latent_dim":   LATENT_DIM,
        "num_classes":  NUM_CLASSES,
    }, path)


def load_checkpoint(model, optimizer, scheduler):
    if not os.path.exists(CKPT_FILE):
        return 0, float("inf")   # start fresh
    ckpt = torch.load(CKPT_FILE, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"], ckpt["best_loss"]


# ─── ONE EPOCH ──────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, kl_weight, train=True):
    model.train() if train else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    tag = "Train" if train else "Val  "

    with ctx:
        bar = tqdm(loader, desc=f"  {tag}", leave=False,
                   dynamic_ncols=True, colour="cyan" if train else "red")
        for x, y in bar:
            x    = x.to(DEVICE, non_blocking=True)
            y_oh = F.one_hot(y.to(DEVICE), NUM_CLASSES).float()

            if train:
                optimizer.zero_grad()

            x_recon, mu, log_var = model(x, y_oh)
            loss = vae_loss(x_recon, x, mu, log_var, kl_weight)

            if train:
                loss.backward()
                optimizer.step()

            bs          = x.size(0)
            total_loss += loss.item()
            total_acc  += pixel_accuracy(x_recon, x) * bs
            n          += bs

            bar.set_postfix(loss=f"{loss.item()/bs:.4f}")

    return total_loss / n, total_acc / n


# ─── GENERATE ───────────────────────────────────────────────────
def generate_digit(model, digit):
    model.eval()
    with torch.no_grad():
        z     = torch.randn(1, LATENT_DIM).to(DEVICE)
        label = F.one_hot(torch.tensor([digit]).to(DEVICE), NUM_CLASSES).float()
        img   = model.generate(z, label)
        save_image(img, f"{SAVE_DIR}/generated_{digit}.png")
    print(f"  Saved digit {digit}")


# ─── MAIN ───────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    transform    = transforms.Compose([transforms.ToTensor()])
    train_data   = datasets.MNIST("./data", train=True,  transform=transform, download=True)
    test_data    = datasets.MNIST("./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model     = CVAE(LATENT_DIM, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── resume if checkpoint exists ─────────────────────────────
    start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler)

    print(f"Device : {DEVICE}")
    print(f"Latent : {LATENT_DIM}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")
    print(f"{'='*75}")

    for epoch in range(start_epoch + 1, EPOCHS + 1):

        kl_weight = min(1.0, epoch / 10.0) if KL_ANNEAL else 1.0

        train_loss, train_acc = run_epoch(model, train_loader, optimizer, kl_weight, train=True)
        test_loss,  test_acc  = run_epoch(model, test_loader,  optimizer, kl_weight, train=False)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:>3}/{EPOCHS}] | "
            f"Train: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
            f"Test: {test_loss:.4f} (Acc: {test_acc:.2f}%) | "
            f"KL_w: {kl_weight:.2f} | "
            f"LR: {current_lr:.6f}"
        )

        # ── save latest every N epochs ──────────────────────────
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, best_loss, CKPT_FILE)
            print(f"  [ckpt] Saved latest → {CKPT_FILE}")

        # ── save best model separately ──────────────────────────
        if test_loss < best_loss:
            best_loss = test_loss
            save_checkpoint(epoch, model, optimizer, scheduler, best_loss, BEST_FILE)
            print(f"  [best] New best test loss {best_loss:.4f} → {BEST_FILE}")

    print(f"\n{'='*75}\nTraining complete. Generating digits...\n")
    for digit in range(10):
        generate_digit(model, digit)
    print(f"\nImages saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()