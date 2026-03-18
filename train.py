import torch
from torch import nn
# Importiamo i "pezzi" dagli altri file
from models.neural_network import NeuralNetworkImm
#se un file python sta dentro una cartella per importarlo devi fare
# cartella.nomefile
from dataset.dataset_loadaer import get_dataloaders
from eval import test
#serve a tenere traccia modello
import wandb


# 1. Setup del device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Inizializzazione modello, caricamento dati, loss e ottimizzatore
model = NeuralNetworkImm().to(device)
train_loader, test_loader = get_dataloaders(batch_size=64)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. La tua funzione di addestramento
def train(epoch, model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return train_accuracy

# 4. Il ciclo di addestramento finale (Il vero e proprio avvio)
#questo pezzo di codice di avvia quando da terminale richiami il file python train.py
# o quando da colab simuli il comando del terminael
if __name__ == "__main__":
    #integro wandb per tenere traccia modello
    #inizializzo
    wandb.init(project = 'lab2')

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train_accuracy = train(epoch, model, train_loader, optimizer, criterion, device)
        # Chiamiamo la funzione test importata dal file eval.py
        test_accuracy = test(model, test_loader, criterion, device)

        #wandb creerà aututomaticamente grafici con dati di
        # questo ciclo for, dati modello
        wandb.log({
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        })

        # SALVIAMO IL CHECKPOINT
        # Creiamo il percorso dove salvare il file (es. checkpoints/modello_epoca_1.pth)
        percorso_salvataggio = f"checkpoints/modello_epoca_{epoch}.pth"
        
        # Salviamo il "cervello" del modello (state_dict)
        torch.save(model.state_dict(), percorso_salvataggio)
        print(f"✅ Checkpoint salvato: {percorso_salvataggio}")