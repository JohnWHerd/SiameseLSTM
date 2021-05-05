from torch import nn
from torch.autograd import Variable
import torch



class Siamese_lstm(nn.Module):
    def __init__(self, embedding):
        super(Siamese_lstm, self).__init__()
        
        #set up dimensions
        self.embed_size = 300
        self.batch_size = 1
        self.hidden_size = 15
        self.num_layers = 1
        self.direction = 1
        
        self.input_dim = 50
        self.inner_dim = 25
        
        self.embedding = embedding
        
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, dropout=0,
                            num_layers=self.num_layers, bidirectional=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 5, self.inner_dim),
            nn.Linear(self.inner_dim, 2)
        )
        
        
    def initHidden(self):
        rand_hid = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        return rand_hid, rand_cell
    
    def encodeWord(self, input, hidden, cell):
        input = self.embedding(input).view(1, 1, -1)
        out, (hidden, cell) = self.lstm(input, (hidden, cell))
        return out, hidden, cell
    
    
    def forward(self, s1, s2):
        
        #get randomized starting states
        h1, c1 = self.initHidden()
        h2, c2 = self.initHidden()
        
        #encode the inputs
        
        for i in range(len(s1)):
            v1, h1, c1 = self.encodeWord(s1[i], h1, c1)
            
        for j in range(len(s2)):
            v2, h2, c2 = self.encodeWord(s2[j], h2, c2)
            
        
        #make our vetor to throw into the classifier
        vec = torch.cat((v1,torch.abs(v1 - v2),v2,v1*v2, (v1+v2)/2), 2)
        
        output = self.classifier(vec)
        
        return output
    
    
class Siamese_lstm_V2(nn.Module):
    def __init__(self, embedding):
        super(Siamese_lstm_V2, self).__init__()
        
        #set up dimensions
        self.embed_size = 300
        self.batch_size = 1
        self.hidden_size = 25
        self.num_layers = 1
        self.direction = 1
        
        self.input_dim = 25
        self.inner_dim = 5
        
        self.embedding = embedding
        
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, dropout=0,
                            num_layers=self.num_layers, bidirectional=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 5, self.inner_dim),
            nn.Linear(self.inner_dim, 2)
        )
        
        
    def initHidden(self):
        rand_hid = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        return rand_hid, rand_cell
    
    def encodeWord(self, input, hidden, cell):
        input = self.embedding(input).view(1, 1, -1)
        out, (hidden, cell) = self.lstm(input, (hidden, cell))
        return out, hidden, cell
    
    
    def forward(self, s1, s2):
        
        #get randomized starting states
        h1, c1 = self.initHidden()
        h2, c2 = self.initHidden()
        
        #encode the inputs
        
        for i in range(len(s1)):
            v1, h1, c1 = self.encodeWord(s1[i], h1, c1)
            
        for j in range(len(s2)):
            v2, h2, c2 = self.encodeWord(s2[j], h2, c2)
            
        
        #make our vetor to throw into the classifier
        vec = torch.cat((v1,torch.abs(v1 - v2),v2,v1*v2, (v1+v2)/2), 2)
        
        output = self.classifier(vec)
        
        return output