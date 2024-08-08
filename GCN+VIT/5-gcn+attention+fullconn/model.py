import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
import config

drop_rate = config.drop_rate


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        attention_weights = F.softmax(self.fc(x), dim=1)
        attended_x = torch.mul(x, attention_weights)
        return attended_x

# used for supervised
class JointlyTrainModel(nn.Module):
    def __init__(self, inchannel, outchannel, batch, testmode=False, **kwargs):
        super(JointlyTrainModel, self).__init__()
        self.batch = batch
        self.testmode = testmode
        linearsize = 512
        outchannel = 128

        # K = [1,2,3,4,5,6,7,8,9,10]
        self.conv1 = gnn.ChebConv(inchannel, outchannel, K=config.K)
        self.dropout1 = nn.Dropout(drop_rate)  # Dropout layer after the first ChebConv layer
        self.conv2 = gnn.ChebConv(outchannel, outchannel, K=config.K)
        self.dropout2 = nn.Dropout(drop_rate)  # Dropout layer after the second ChebConv layer
        self.conv3 = gnn.ChebConv(outchannel, outchannel, K=config.K)
        self.dropout3 = nn.Dropout(drop_rate)  # Dropout layer after the third ChebConv layer
        self.conv4 = gnn.ChebConv(outchannel, outchannel, K=config.K)
        self.dropout4 = nn.Dropout(drop_rate)  # Dropout layer after the fourth ChebConv layer
        self.conv5 = gnn.ChebConv(outchannel, outchannel, K=config.K)
        self.dropout5 = nn.Dropout(drop_rate)  # Dropout layer after the fifth ChebConv layer
        self.relu = nn.ReLU(inplace=True)
        self.attention = AttentionLayer(5 * outchannel, 5 * outchannel)

        self.HC = nn.Sequential(
            nn.Linear(62 * 5 * outchannel, linearsize),
            nn.BatchNorm1d(linearsize),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize, linearsize // 2),
            nn.BatchNorm1d(linearsize // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize // 2, kwargs['HC'])
        )

    def forward(self, *args):
        if not self.testmode:
            x, e = args[0].x, args[0].edge_index  # original graph data
            x1 = self.relu(self.conv1(x, e))
            x1 = self.dropout1(x1)
            x2 = self.relu(self.conv2(x1, e))
            # x2 = self.dropout2(x2)
            x3 = self.relu(self.conv3(x2, e))
            x4 = self.relu(self.conv4(x3, e))
            # x4 = self.dropout4(x4)
            x5 = self.relu(self.conv5(x4, e))
            x1_to_x5 = torch.cat([x1, x2, x3, x4, x5], dim=1)
            attended_x = self.attention(x1_to_x5)
            attended_x = attended_x.view(self.batch, -1)
            output = self.HC(attended_x)
            output = F.softmax(output, dim=1)
            return output
        else:
            x, e = args[0].x, args[0].edge_index  # original graph data
            x1 = self.relu(self.conv1(x, e))
            x1 = self.dropout1(x1)
            x2 = self.relu(self.conv2(x1, e))
            x2 = self.dropout2(x2)
            x3 = self.relu(self.conv3(x2, e))
            x3 = self.dropout3(x3)
            x4 = self.relu(self.conv4(x3, e))
            x4 = self.dropout4(x4)
            x5 = self.relu(self.conv5(x4, e))
            x5 = self.dropout5(x5)
            x1_to_x5 = torch.cat([x1, x2, x3, x4, x5], dim=1)
            attended_x = self.attention(x1_to_x5)
            attended_x = attended_x.view(self.batch, -1)
            output = self.HC(attended_x)
            output = F.softmax(output, dim=1)
            return output


