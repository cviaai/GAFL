from tqdm import tqdm
import torch


def run_epoch(model, iterator,
              criterion, optimizer,
              metrics,
              phase='train', epoch=0,
              device='cpu', writer=None):
    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_metrics = dict((metric_name, 0.0) for metric_name in metrics.keys())

    with torch.set_grad_enabled(is_train):
        for (images, targets) in tqdm(iterator, desc=f"{phase}", ascii=True):
            images, targets = images.to(device), targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            for metric_name in metrics.keys():
                epoch_metrics[metric_name] += metrics[metric_name](predictions.detach(), targets)

        epoch_loss /= len(iterator)
        for metric_name in metrics.keys():
            epoch_metrics[metric_name] /= len(iterator)

        if writer is not None:
            writer.add_scalar(f"loss/{phase}", epoch_loss, epoch)
            for metric_name in metrics.keys():
                writer.add_scalar(f"{metric_name}/{phase}", epoch_metrics[metric_name], epoch)

        return epoch_loss, epoch_metrics


def print_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics):
    print(f'Epoch: {epoch + 1:02}')

    print(f'\tTrain Loss: {train_loss:.2f} | Train Metrics: ' +
          ' | '.join([metric_name + ': ' + f"{train_metrics[metric_name]:.2f}"
                      for metric_name in train_metrics.keys()]))

    print(f'\t  Val Loss: {val_loss:.2f} |   Val Metrics: ' +
          ' | '.join([metric_name + ': ' + f"{val_metrics[metric_name]:.2f}"
                      for metric_name in val_metrics.keys()]))
    
    
def train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metrics,
          n_epochs,
          device,
          writer):
    best_val_loss = float('+inf')
    for epoch in range(n_epochs):
        train_loss, train_metrics = run_epoch(model, train_dataloader,
                                              criterion, optimizer,
                                              metrics,
                                              phase='train', epoch=epoch,
                                              device=device, writer=writer)
        val_loss, val_metrics = run_epoch(model, val_dataloader,
                                          criterion, None,
                                          metrics,
                                          phase='val', epoch=epoch,
                                          device=device, writer=writer)
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{model.name}.best.pth")

        print_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics)

    if writer is not None:
        writer.close()
