from __future__ import print_function,division

import sys
import numpy as np
import h5py

import torch


class Uniprot21:
    def __init__(self):
        missing = 20
        self.chars = np.frombuffer(b'ARNDCQEGHILKMFPSTWYVXOUBZ', dtype=np.uint8)

        self.encoding = np.zeros(256, dtype=np.uint8) + missing

        encoding = np.arange(len(self.chars))
        encoding[21:] = [11,4,20,20] # encode 'OUBZ' as synonyms

        self.encoding[self.chars] = encoding
        self.size = encoding.max() + 1
        self.mask = False

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """ encode a byte string into alphabet indices """
        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """ decode index array, x, to byte string of this alphabet """
        string = self.chars[x]
        return string.tobytes()

    def unpack(self, h, k):
        """ unpack integer h into array of this alphabet with length k """
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """ retrieve byte string of length k decoded from integer h """
        kmer = self.unpack(h, k)
        return self.decode(kmer)





from argparse import Namespace
args = Namespace(model = 'prose_mt',path='./',device=0
        )

# load the model
if args.model == 'prose_mt':
    from prose.models.multitask import ProSEMT
    print('# loading the pre-trained ProSE MT model', file=sys.stderr)
    model = ProSEMT.load_pretrained()
elif args.model == 'prose_dlm':
    from prose.models.lstm import SkipLSTM
    print('# loading the pre-trained ProSE DLM model', file=sys.stderr)
    model = SkipLSTM.load_pretrained()
else:
    print('# loading model:', args.model, file=sys.stderr)
    model = torch.load(args.model)
model.eval()
model_state = model.state_dict()
sequence = b'MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPVECPKAPVEWNNPPS'
use_cuda = True
model = model.cuda()
if len(sequence) == 0:
    n = model.embedding.proj.weight.size(1)
    z = np.zeros((1,n), dtype=np.float32)

else:
    alphabet = Uniprot21()
    sequence = sequence.upper()
    # convert to alphabet index
    sequence = alphabet.encode(sequence)
    sequence = torch.from_numpy(sequence)
    if use_cuda:
        sequence = sequence.cuda()

    # embed the sequence
    with torch.no_grad():
        sequence = sequence.long().unsqueeze(0)
        z = model.transform(sequence)
        # pool if needed
        z = z.squeeze(0)
        z = z.cpu().numpy()
        print(z.shape)








import sys
sys.exit()

# set the device
d = args.device
use_cuda = (d != -1) and torch.cuda.is_available()
if d >= 0:
    torch.cuda.set_device(d)

if use_cuda:
    model = model.cuda()

# parse the sequences and embed them
# write them to hdf5 file
print('# writing:', args.output, file=sys.stderr)
h5 = h5py.File(args.output, 'w')

pool = args.pool
print('# embedding with pool={}'.format(pool), file=sys.stderr)
count = 0
with open(path, 'rb') as f:
    for name,sequence in fasta.parse_stream(f):
        pid = name.decode('utf-8')
        z = embed_sequence(model, sequence, pool=pool, use_cuda=use_cuda)
        # write as hdf5 dataset
        h5.create_dataset(pid, data=z)
        count += 1
        print('# {} sequences processed...'.format(count), file=sys.stderr, end='\r')
print(' '*80, file=sys.stderr, end='\r')


