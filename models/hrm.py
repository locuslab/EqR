from models.trm import Block, Blocks, InnerNetwork as TRMInnerNetwork, TRMModel


class InnerNetwork(TRMInnerNetwork):
    def _init_levels(self, config):
        super()._init_levels(config)
        self.H_level = Blocks([Block(config) for _ in range(config.H_layers)])

    def latent_recursion(self, z_H, z_L, x, seq):
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + x, **seq)
        return self.H_level(z_H, z_L, **seq), z_L


class HRMModel(TRMModel):
    inner_class = InnerNetwork
