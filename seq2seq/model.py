import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

MAX_LENGTH = 128
SOS_token = 0
EOS_token = 1

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    

class Decoder(nn.Module):
  def __init__(self,hidden_size, output_size):
      super(Decoder, self).__init__()
      self.embedding = nn.Embedding(output_size, hidden_size)
      self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
      self.fc_out = nn.Linear(hidden_size, output_size)
  def forward(self,input,hidden ):
      output_emb = self.embedding(input)
      output_relu = F.relu(output_emb)
      output_gru, hidden = self.gru(output_relu,hidden)
      output = self.fc_out(output_gru)
      return output, hidden
  

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  def forward(self, input_tensor,target_tensor = None):
      encoder_output, encoder_hidden = self.encoder(input_tensor)
      batch_size = encoder_output.size(0)
      decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)
      decoder_hidden = encoder_hidden
      decoder_outputs = []

      for i in range(MAX_LENGTH):
          decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden)
          decoder_outputs.append(decoder_output)
          if target_tensor is not None:

            decoder_input = target_tensor[:, i].unsqueeze(1)
          else:
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach()
      decoder_outputs = torch.cat(decoder_outputs, dim=1)
      decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
      return decoder_outputs, decoder_hidden, None