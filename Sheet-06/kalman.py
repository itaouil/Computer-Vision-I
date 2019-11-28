import numpy as np
import matplotlib.pylab as plt

observations = np.load('observations.npy')


def get_observation(t):
    return observations[t]


class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):
        self.psi = psi
        self.sigma_p = sigma_p
        self.phi = phi
        self.sigma_m = sigma_m
        self.states = np.zeros((4, (tau + 1)))
        self.cov = None
        self.tau = tau

    def init(self, init_state):
        self.states[:, 0] = init_state
        self.cov = np.identity(4)

    def track(self, xt):
        xt = np.array([xt]).T
        cov_p = self.compute_tau_pred()
        self.compute_correction(cov_p, xt)

        return False

    def compute_tau_pred(self):
        state_t_1 = self.states[:, 0]
        state_t_p = np.dot(self.psi, state_t_1)
        self.states = np.roll(self.states, 1)
        self.states[:, 0] = state_t_p
        cov_p = self.sigma_p + np.dot(np.dot(self.psi, self.cov), self.psi.T)
        return cov_p

    def compute_correction(self, cov_p, xt):
        k_gain = np.dot(
            np.dot(cov_p, self.phi.T),
            # ----
            np.linalg.inv(
                (self.sigma_m + np.dot(np.dot(self.phi, cov_p), self.phi.T))
            )
        )

        self.states = self.states + np.dot(k_gain, (xt - np.dot(self.phi, self.states)))
        self.cov = np.dot((np.identity(4) - np.dot(k_gain, self.phi)), cov_p)

    def get_current_location(self):
        return self.states[:, -1]


def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())
    return track


def main():
    init_state = np.array([0, 1, 0, 0])

    psi = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    phi = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
    sm = 0.05
    sigma_m = np.array([[sm, 0],
                        [0, sm]])

    tracker = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=0)
    tracker.init(init_state)

    fixed_lag_smoother = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=25)
    fixed_lag_smoother.init(init_state)

    track = perform_tracking(tracker)
    track_smoothed = perform_tracking(fixed_lag_smoother)

    plt.figure()
    # plt.plot([x[0] for x in observations], [x[1] for x in observations])
    # plt.plot([x[0] for x in track], [x[1] for x in track])
    plt.plot([x[0] for x in track_smoothed], [x[1] for x in track_smoothed])

    plt.show()


if __name__ == "__main__":
    main()
