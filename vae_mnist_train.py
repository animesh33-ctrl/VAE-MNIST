import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 256
EPOCHS      = 100           # epochs THIS run (resumes from checkpoint if exists)
LR          = 1e-3
LATENT_DIM  = 20
SAVE_DIR    = "vae_outputs"
RESUME      = True         # set False to train from scratch

# ─── ENCODER 
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28→14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14→7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),            # 7→7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),                                # 128*7*7 = 6272
        )
        with torch.no_grad():
            flat_size = self.conv(torch.zeros(1, 1, 28, 28)).shape[1]

        self.fc_mu      = nn.Linear(flat_size, latent_dim)
        self.fc_log_var = nn.Linear(flat_size, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc_mu(h), self.fc_log_var(h)


# ─── DECODER ──────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),                            # 7→7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 7→14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14→28
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 128, 7, 7)
        return self.deconv(h)


# ─── VAE ──────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def decode(self, z):
        return self.decoder(z)


# ─── LOSS ─────────────────────────────────────────────────────────────────────
def vae_loss(x_recon, x, mu, log_var, kl_weight):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_weight * kl_loss


# ─── FIXED TEST BATCH for consistent reconstruction visualization ──────────────
def get_fixed_test_batch(test_dataset, device, n=32):
    """Always same 32 images — not dataloader dependent."""
    imgs = test_dataset.data[:n].unsqueeze(1).float().div(255).to(device)
    return imgs


# ─── SAVE SAMPLES ─────────────────────────────────────────────────────────────
def save_samples(model, epoch, test_dataset, device, save_dir, latent_dim):
    model.eval()
    with torch.no_grad():
        # 1. Generated from random z
        z       = torch.randn(64, latent_dim).to(device)
        samples = model.decode(z)
        save_image(samples, f"{save_dir}/generated_epoch_{epoch:03d}.png", nrow=8)

        # 2. Reconstruction — fixed batch, consistent across epochs ✅
        x_fixed       = get_fixed_test_batch(test_dataset, device)
        x_recon, _, _ = model(x_fixed)
        comparison    = torch.cat([x_fixed, x_recon])
        save_image(comparison, f"{save_dir}/reconstruction_epoch_{epoch:03d}.png", nrow=8)


# ─── INTERPOLATION: encoded real digits, not random z ─────────────────────────
def save_interpolations(model, test_dataset, device, save_dir, latent_dim):
    model.eval()
    pairs = [(3, 7), (1, 9), (0, 8), (2, 5)]  # interesting morph pairs

    with torch.no_grad():
        for (d1, d2) in pairs:
            labels = test_dataset.targets
            idx1   = (labels == d1).nonzero(as_tuple=True)[0][0].item()
            idx2   = (labels == d2).nonzero(as_tuple=True)[0][0].item()

            x1 = test_dataset.data[idx1].unsqueeze(0).unsqueeze(0).float().div(255).to(device)
            x2 = test_dataset.data[idx2].unsqueeze(0).unsqueeze(0).float().div(255).to(device)

            mu1, _ = model.encoder(x1)
            mu2, _ = model.encoder(x2)

            # 10-step interpolation between mu1 and mu2
            imgs = [model.decode((1 - a) * mu1 + a * mu2)
                    for a in torch.linspace(0, 1, 10).to(device)]
            save_image(torch.cat(imgs),
                       f"{save_dir}/interp_{d1}to{d2}.png", nrow=10)
            print(f"  Saved interp_{d1}to{d2}.png")

        # Also save random z interpolation for comparison
        z1, z2 = torch.randn(1, latent_dim).to(device), torch.randn(1, latent_dim).to(device)
        imgs   = [model.decode((1 - a) * z1 + a * z2)
                  for a in torch.linspace(0, 1, 10).to(device)]
        save_image(torch.cat(imgs), f"{save_dir}/interp_random_z.png", nrow=10)
        print(f"  Saved interp_random_z.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    transform     = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True,  transform=transform, download=True)
    test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True, prefetch_factor=2)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True, prefetch_factor=2)

    model     = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── RESUME from checkpoint ────────────────────────────────────────────────
    start_epoch    = 1
    best_test_loss = float("inf")
    ckpt_path      = f"{SAVE_DIR}/vae_best.pth"

    if RESUME and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # ✅ restore optimizer too
        start_epoch    = ckpt["epoch"] + 1
        best_test_loss = ckpt["test_loss"]
        print(f"Resumed from epoch {ckpt['epoch']} | best loss: {best_test_loss:.4f}")
    else:
        print("Training from scratch.")

    end_epoch = start_epoch + EPOCHS
    # CosineAnnealing over THIS run's epochs — not reset to 0 ✅
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training epochs {start_epoch} → {end_epoch - 1}")
    print("=" * 60)

    # ── TRAIN LOOP ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, end_epoch):
        # KL annealing: ramp over first 30 epochs of TOTAL training
        kl_weight = min(1.0, epoch / 30)

        # Train
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            x_recon, mu, log_var = model(x)
            loss = vae_loss(x_recon, x, mu, log_var, kl_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)

        # Test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(DEVICE, non_blocking=True)
                x_recon, mu, log_var = model(x)
                test_loss += vae_loss(x_recon, x, mu, log_var, kl_weight=1.0).item()
        test_loss /= len(test_loader.dataset)

        scheduler.step()

        print(f"Epoch [{epoch:>4}/{end_epoch - 1}] | Train: {train_loss:.4f} | Test: {test_loss:.4f} "
              f"| KL_w: {kl_weight:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save images every 5 epochs
        if epoch % 5 == 0:
            save_samples(model, epoch, test_dataset, DEVICE, SAVE_DIR, LATENT_DIM)

        # Save best checkpoint
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),  # ✅ save optimizer
                "test_loss":            test_loss,
                "latent_dim":           LATENT_DIM,
            }, ckpt_path)

    print(f"\nDone! Best test loss: {best_test_loss:.4f}")

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    print("\n--- Loading best model for inference ---")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Saving interpolations (real encoded digits)...")
    save_interpolations(model, test_dataset, DEVICE, SAVE_DIR, LATENT_DIM)

    # Final generation grid
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, LATENT_DIM).to(DEVICE)
        save_image(model.decode(z), f"{SAVE_DIR}/final_generated.png", nrow=8)
        print(f"Saved final_generated.png")

    print(f"\nAll outputs → ./{SAVE_DIR}/")


if __name__ == "__main__":
    main()