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






