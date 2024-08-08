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

        outchannel=62


        # K = [1,2,3,4,5,6,7,8,9,10]
        self.conv1 = gnn.ChebConv(inchannel, outchannel, K=config.K)
        self.conv2 = gnn.ChebConv(inchannel+outchannel, outchannel, K=config.K)
        self.conv3 = gnn.ChebConv(inchannel+2*outchannel, outchannel, K=config.K)
        self.conv4 = gnn.ChebConv(inchannel+3*outchannel, outchannel, K=config.K)
        # self.conv5 = gnn.ChebConv(inchannel+4*outchannel, outchannel, K=config.K)
        self.relu = nn.ReLU(inplace=True)
        # self.attention = AttentionLayer(4 * outchannel, 4 * outchannel)
        # self.attention = AttentionLayer(5 * outchannel, 5 * outchannel)

        self.HC = nn.Sequential(
            # nn.Linear(outchannel * 62, linearsize),
            # nn.Linear(62*5*outchannel, linearsize),
            nn.Linear(62 * 4 * outchannel, linearsize),
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

            # x1 = F.relu(self.conv1(x, e))
            # x1 = F.relu(self.conv2(x1, e))
            # x1 = F.relu(self.conv3(x1, e))
            # x1 = F.relu(self.conv4(x1, e))
            # x1 = F.relu(self.conv5(x1, e))

            x1 = self.relu(self.conv1(x, e))
            x2 = self.relu(self.conv2(torch.cat([x, x1], 1),e))
            x3 = self.relu(self.conv3(torch.cat([x, x1, x2], 1),e))
            x4 = self.relu(self.conv4(torch.cat([x, x1, x2, x3], 1),e))
            # x5 = self.relu(self.conv5(torch.cat([x, x1, x2, x3, x4], 1),e))
            # x1_to_x5 = torch.cat([x1, x2, x3, x4, x5], dim=1)
            x1_to_x5 = torch.cat([x1, x2, x3, x4], dim=1)
            attended_x = self.attention(x1_to_x5)
            attended_x = attended_x.view(self.batch, -1)
            output = self.HC(attended_x)
            output = F.softmax(output, dim=1)
            # x5 = x5.view(self.batch, -1)
            # x5 = self.HC(x5)
            # x5 = F.softmax(x5, dim=1)
            return output

        else:
            xx, e3 = args[0].x, args[0].edge_index  # original graph data
            x1 = self.relu(self.conv1(xx, e3))
            x2 = self.relu(self.conv2(torch.cat([xx, x1], 1),e3))
            x3 = self.relu(self.conv3(torch.cat([xx, x1, x2], 1),e3))
            x4 = self.relu(self.conv4(torch.cat([xx, x1, x2, x3], 1),e3))
            # x5 = self.relu(self.conv5(torch.cat([xx, x1, x2, x3, x4], 1),e3))
            # x1_to_x5 = torch.cat([x1, x2, x3, x4, x5], dim=1)
            x1_to_x5 = torch.cat([x1, x2, x3, x4], dim=1)
            attended_x = self.attention(x1_to_x5)
            attended_x = attended_x.view(self.batch, -1)
            output = self.HC(attended_x)
            output = F.softmax(output, dim=1)
            return output
            # x5 = torch.cat([x1, x2, x3, x4, x5], dim=1)
            # x3 = x5.view(self.batch, -1)
            # x3 = self.HC(x3)
            # x3 = F.softmax(x3, dim=1)
            # return x3

