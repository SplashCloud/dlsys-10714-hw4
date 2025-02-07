import sys
sys.path.append('./python')
sys.path.append('./apps')
import needle as ndl
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cpu()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=256, device=device, dtype="float32")
model = LanguageModel(20, len(corpus.dictionary), hidden_size=32, num_layers=1, seq_model='transformer', seq_len=20, device=device)
train_acc, train_loss = train_ptb(model, train_data, seq_len=20, n_epochs=10, device=device, lr=0.003, optimizer=ndl.optim.Adam)
print(f'train_acc={train_acc}, train_loss={train_loss}')
test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=20, device=device)
print(f'test_acc={test_acc}, train_loss={train_loss}')
# max_lines = 1000
# train_acc=0.2371124267578125, train_loss=0.0008183120240573772
# test_acc=0.2341888427734375, train_loss=0.0008183120240573772