from Blocks import *

class Generator(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):

        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)

def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):


    fake_Y = gen_XY(real_X)
    disc_pred = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(disc_pred, torch.ones_like(disc_pred))
    return adversarial_loss, fake_Y

def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):

    adversarial_loss_A, fake_B = get_gen_adversarial_loss(real_A,disc_B,gen_AB,adv_criterion)
    adversarial_loss_B , fake_A= get_gen_adversarial_loss(real_B, disc_A,gen_BA, adv_criterion)
    adversarial_loss = adversarial_loss_A + adversarial_loss_B

    identity_loss_A, id_A = get_identity_loss(real_A,gen_BA, identity_criterion)
    identity_loss_B , id_B= get_identity_loss(real_B, gen_AB, identity_criterion)
    identity_loss = identity_loss_A + identity_loss_B

    cycle_consisteny_loss_A, cy_A = get_cycle_consistency_loss(real_A,fake_B, gen_BA,cycle_criterion)
    cycle_consisteny_loss_B, cy_B = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    cycle_consisteny_loss = cycle_consisteny_loss_A + cycle_consisteny_loss_B
    gen_loss = lambda_identity * identity_loss +  lambda_cycle * cycle_consisteny_loss + adversarial_loss

    return gen_loss, fake_A, fake_B