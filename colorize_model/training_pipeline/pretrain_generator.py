from tqdm import tqdm, trange
import torch


def pretrain_generator(generator, train_dataloader, optimizer, criterion, epochs, path, device):
    losses = []
    for epoch in trange(1, epochs + 1):
        current_loss = 0
        for img_l, img_ab in tqdm(train_dataloader, leave=False):
            img_l, img_ab = img_l.to(device), img_ab.to(device)
            preds = generator(img_l)
            loss = criterion(preds, img_ab)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            current_loss += loss.detach().cpu().mean().item()

        losses.append(current_loss / len(train_dataloader))
        print(f"Epoch {epoch}/{epochs}: L1 Loss = {round(losses[-1], 5)}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'l1_loss': losses[-1],
        }, path)
    return losses
