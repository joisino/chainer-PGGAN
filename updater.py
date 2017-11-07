import numpy as np
import chainer
import chainer.functions as F
from chainer import reporter
from chainer import Variable

class WganGpUpdater(chainer.training.StandardUpdater):
    def __init__(self, alpha, delta, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.lam = 10
        self.epsilon_drift = 0.001
        self.alpha = alpha
        self.delta = delta
        super(WganGpUpdater, self).__init__(**kwargs)

    def update_core(self):
        opt_g = self.get_optimizer('gen')
        opt_d = self.get_optimizer('dis')

        xp = self.gen.xp

        # update discriminator
        x = self.get_iterator('main').next()
        x = xp.array(x)
        m = len(x)

        z = self.gen.z(m)
        x_tilde = self.gen(z, self.alpha).data
        
        epsilon = xp.random.rand(m, 1, 1, 1).astype('f')
        x_hat = Variable(epsilon * x + (1 - epsilon) * x_tilde)

        dis_x = self.dis(x, self.alpha)
        
        loss_d = self.dis(x_tilde, self.alpha) - dis_x

        g_d, = chainer.grad([self.dis(x_hat, self.alpha)], [x_hat], enable_double_backprop=True)
        g_d_norm = F.sqrt(F.batch_l2_norm_squared(g_d) + 1e-6)
        g_d_norm_delta = g_d_norm - 1
        loss_l = self.lam * g_d_norm_delta * g_d_norm_delta
        
        loss_dr = self.epsilon_drift * dis_x * dis_x

        dis_loss = F.mean(loss_d + loss_l + loss_dr)

        self.dis.cleargrads()
        dis_loss.backward()
        opt_d.update()
        
        # update generator
        z = self.gen.z(m)
        x = self.gen(z, self.alpha)
        gen_loss = F.average(-self.dis(x, self.alpha))

        self.gen.cleargrads()
        gen_loss.backward()
        opt_g.update()

        reporter.report({'loss_d': F.mean(loss_d), 'loss_l': F.mean(loss_l), 'loss_dr': F.mean(loss_dr), 'dis_loss': dis_loss, 'gen_loss': gen_loss, 'alpha': self.alpha})

        self.alpha = self.alpha + self.delta
