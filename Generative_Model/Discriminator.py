from Blocks import *

class Discriminator(nn.Module):

    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn
    
def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):

    disc_pred1 = disc_X(real_X)
    disc_loss1 = adv_criterion(disc_pred1, torch.ones_like(disc_pred1))
    disc_pred2 =disc_X(fake_X.detach())
    disc_loss2 =adv_criterion(disc_pred2, torch.zeros_like(disc_pred2))
    disc_loss = (disc_loss1 + disc_loss2)/2
    return disc_loss