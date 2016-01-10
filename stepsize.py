
class StepSizeGenerator(object):

    def __init__(self, max_epoch, eps_start=0.01, eps_end=0.0001, gamma=0.55):
        self.a, self.b = self.__calc_ab(eps_start, eps_end, gamma, max_epoch)
        self.gamma = gamma

    def __calc_ab(self, eps_start, eps_end, gamma, epoch):
        """Returns coefficients that characterize step size

        Args:
            eps_start(float): initial step size
            eps_end(float): initial step size
            gamma(float): decay rate
            epoch(int): # of epoch
        Returns:
            pair of float: (A, B) satisfies ``A / B ** gamma == eps_start``
            and ``A / (B + epoch) ** gamma == eps_end``
        """

        B = 1 / ((eps_start / eps_end) ** (1 / gamma) - 1) * epoch
        A = eps_start * B ** gamma
        eps_start_actual = A / B ** gamma
        eps_end_actual = A / (B + epoch) ** gamma
        assert abs(eps_start - eps_start_actual) < 1e-4
        assert abs(eps_end - eps_end_actual) < 1e-4
        return A, B

    def __call__(self, epoch):
        return self.a / (self.b + epoch) ** self.gamma
