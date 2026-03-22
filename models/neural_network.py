#Definisco quindi la classe per la NeuralNetwork per immagini
import torch
from torch import nn
import torchvision

class NeuralNetworkImm(nn.Module):
  def __init__(self):
    super(NeuralNetworkImm, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding = 1)
    #padding = 1 serve a non perdere un po di dimensione
    #Nota applicativa: siccome a ogni passaggio diminuiamo di 4 la dim
    # dell'immagine con il pool ha senso raddoppiare il numero di nodi
    # kernel (per cercare meglio intuitivamente). Questo è lo standard.
    # Nota ulteriore questo non crea colli di bottiglia
    self.conv2 = nn.Conv2d(64,128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(128,256, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2,2) #estrae massimo da sottomatrici 2x2
    self.relu = nn.ReLU()
    #IMPORTANTE: ultimo layer lineare per problemi di classificazione
    self.last_layer = nn.Linear(256*8*8, 200) #200 è il numero di label (classi)
    #restiuisce logits per ogni classe

  def forward(self, x):
      #1 strato
      #L'immagine va da 64 a 32 pixel
      x = self.conv1(x)
      x = self.relu(x)
      x = self.pool(x)

      #2 strato
      x = self.pool(self.relu(self.conv2(x)))

      #3 strato
      x = self.pool(self.relu(self.conv3(x)))

      #flattening torch.flatten(x,1) significa
      #Non toccare la dimensione 0 (il Numero di Immagini).
      #Tieni ogni immagine separata. Ma dalla dimensione 1 in poi (Canali, Altezza, Larghezza), srotola tutto in un'unica fila
      x = torch.flatten(x,1)

      logits = self.last_layer(x)
      return logits
  
class NewNeuralNetwork(nn.Module):
    def __init__(self):
      super().__init__()
      # Blocco 1: 3 → 64 canali
      self.conv1_1 = nn.Conv2d(in_channels=3,   out_channels=64,  kernel_size=3, padding=1)
      self.bn1_1   = nn.BatchNorm2d(64)
      self.conv1_2 = nn.Conv2d(in_channels=64,  out_channels=64,  kernel_size=3, padding=1)
      self.bn1_2   = nn.BatchNorm2d(64)

      # Blocco 2: 64 → 128 canali
      self.conv2_1 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
      self.bn2_1   = nn.BatchNorm2d(128)
      self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
      self.bn2_2   = nn.BatchNorm2d(128)

      # Blocco 3: 128 → 256 canali
      self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
      self.bn3_1   = nn.BatchNorm2d(256)
      self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
      self.bn3_2   = nn.BatchNorm2d(256)

      self.relu        = nn.ReLU()
      self.pool        = nn.MaxPool2d(kernel_size=2, stride=2)
      #global pooling, comodo per evitare di calcolare dimensioni output
      self.globalPool  = nn.AdaptiveAvgPool2d((1,1))
      self.lastLayer   = nn.Linear(256, 200)

    def forward(self, x):
      # Blocco 1 — pool solo alla fine del blocco, non dopo ogni conv
      x = self.relu(self.bn1_1(self.conv1_1(x)))
      x = self.pool(self.relu(self.bn1_2(self.conv1_2(x))))

      # Blocco 2
      x = self.relu(self.bn2_1(self.conv2_1(x)))
      x = self.pool(self.relu(self.bn2_2(self.conv2_2(x))))

      # Blocco 3
      x = self.relu(self.bn3_1(self.conv3_1(x)))
      x = self.pool(self.relu(self.bn3_2(self.conv3_2(x))))

      x = self.globalPool(x)
      x = torch.flatten(x, 1)
      logits = self.lastLayer(x)
      return logits







