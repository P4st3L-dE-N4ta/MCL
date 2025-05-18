import numpy as np
import matplotlib.pyplot as plt
import yaml
import re
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
class Map:
    def read_pgm_file(self, file_path, byteorder='>'):
        with open(file_path, 'rb') as file:
            buffer = file.read()
        try:
            header, self.width, self.height, max_val = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:[\r\n])*)", buffer).groups()
        except AttributeError:
            raise ValueError("Invalid PGM file")

        self.width = int(self.width)
        self.height = int(self.height)

        return np.frombuffer(
            buffer,
            dtype='u1' if int(max_val) < 256 else byteorder + 'u2',
            count=self.width * self.height,
            offset=len(header)
        ).reshape((self.height, self.width))

    def read_yaml_file(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        self.resolution = data['resolution']
        self.origin = data['origin']

    def pixel_position(self, x, y):
        a = ((x - self.origin[0]) / self.resolution).astype(int)
        x_pix = np.clip(a, 0, self.width - 1)
        b = (self.height - ((y - self.origin[1]) / self.resolution)).astype(int)
        y_pix = np.clip(b, 0, self.height - 1)
        return x_pix, y_pix

    def valid_position(self, x, y):
        x_pix, y_pix = self.pixel_position(np.array([x]), np.array([y]))
        return self.img[y_pix[0], x_pix[0]] > 250

class LaserSensor:
    def __init__(self, map_obj, laser_max_range=3.5):
        self.map_img = map_obj.img
        self.resolution = map_obj.resolution
        self.origin = map_obj.origin
        self.width = map_obj.width
        self.height = map_obj.height
        self.laser_max_range = laser_max_range

    def pixel_position(self, x, y):
        a = ((x - self.origin[0]) / self.resolution).astype(int)
        x_pix = np.clip(a, 0, self.width - 1)
        b = (self.height - ((y - self.origin[1]) / self.resolution)).astype(int)
        y_pix = np.clip(b, 0, self.height - 1)
        return x_pix, y_pix

    def raycasting(self, x, y, theta):
        angles = np.radians(np.arange(0, 360, 3))
        phi_angles = theta + angles
        range_values = np.arange(0, self.laser_max_range, self.resolution)

        x_coords = x + np.outer(range_values, np.cos(phi_angles))
        y_coords = y + np.outer(range_values, np.sin(phi_angles))

        x_pixels, y_pixels = self.pixel_position(x_coords.flatten(), y_coords.flatten())
        x_pixels = x_pixels.reshape(len(range_values), len(angles))
        y_pixels = y_pixels.reshape(len(range_values), len(angles))

        mask = (self.map_img[y_pixels, x_pixels] < 250)
        hit_indices = np.argmax(mask, axis=0)

        dists = []
        for i, idx in enumerate(hit_indices):
            d = range_values[idx] if idx > 0 else self.laser_max_range
            dists.append(d)
        return np.array(dists)

class Pose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

def sample(variance):
    return np.random.normal(0, variance)

def sample_motion_model_odometry(u_t, x_prev, alpha):
    x_prev_bar, x_curr_bar = u_t

    delta_rot_1 = math.atan2(x_curr_bar.y - x_prev_bar.y, x_curr_bar.x - x_prev_bar.x) - x_prev_bar.theta
    delta_rot_1 = (delta_rot_1 + np.pi) % (2 * np.pi) - np.pi

    delta_trans = math.hypot(x_prev_bar.x - x_curr_bar.x, x_prev_bar.y - x_curr_bar.y)
    delta_rot_2 = x_curr_bar.theta - x_prev_bar.theta - delta_rot_1
    delta_rot_2 = (delta_rot_2 + np.pi) % (2 * np.pi) - np.pi

    delta_rot1_hat = delta_rot_1 - sample(alpha[0]*(delta_rot_1)**2 + alpha[1]*delta_trans**2)
    delta_trans_hat = delta_trans - sample(alpha[2]*delta_trans**2 + alpha[3]*((delta_rot_1)**2+(delta_rot_2)**2))
    delta_rot2_hat = delta_rot_2 - sample(alpha[0]*(delta_rot_2)**2 + alpha[1]*delta_trans**2)

    x_new = x_prev.x + delta_trans_hat * math.cos(x_prev.theta + delta_rot1_hat)
    y_new = x_prev.y + delta_trans_hat * math.sin(x_prev.theta + delta_rot1_hat)
    theta_new = x_prev.theta + delta_rot1_hat + delta_rot2_hat

    # Normalize angle to [-pi, pi]
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

    return (x_new, y_new, theta_new)


def beam_range_finder_model(z, z_star, z_max, sigma_hit=0.2, lambda_short=0.1, z_hit=0.7, z_short=0.1, z_max_weight=0.1, z_rand=0.1):
    z = np.asarray(z)
    z_exp = np.asarray(z_star)
    assert z.shape == z_exp.shape, "z and z_expected must have the same shape"
    total_w = z_hit + z_short + z_max_weight + z_rand
    assert np.isclose(total_w, 1.0), f"Weights sum to {total_w:.3f}, but must sum to 1.0"

    p_hit_unnorm = norm.pdf(z, loc=z_exp, scale=sigma_hit)
    cdf_low  = norm.cdf(0,    loc=z_exp, scale=sigma_hit)
    cdf_high = norm.cdf(z_max, loc=z_exp, scale=sigma_hit)
    eta_hit = cdf_high - cdf_low
    p_hit = np.where((z >= 0) & (z <= z_max), p_hit_unnorm / eta_hit, 0.0)

    p_short_unnorm = np.where((z >= 0) & (z <= z_exp), lambda_short * np.exp(-lambda_short * z), 0.0)
    eta_short = 1.0 - np.exp(-lambda_short * z_exp)
    p_short = np.where(eta_short > 0, p_short_unnorm / eta_short, 0.0)

    p_max = np.where(z == z_max, 1.0, 0.0)
    p_rand = np.where((z >= 0) & (z < z_max), 1.0 / z_max, 0.0)

    p_mix = (z_hit* p_hit + z_short* p_short + z_max_weight * p_max + z_rand* p_rand)

    q = np.prod(p_mix)
    return q



def initialize_map_and_laser(map_file="lsdc4_map.pgm", yaml_file="lsdc4_map.yaml"):
    map_obj = Map()
    map_obj.img = map_obj.read_pgm_file(map_file)
    map_obj.read_yaml_file(yaml_file)
    laser = LaserSensor(map_obj)
    return map_obj, laser

def random_valid_pose(map_obj):
    while True:
        x = np.random.uniform(0, map_obj.width * map_obj.resolution)
        y = np.random.uniform(0, map_obj.height * map_obj.resolution)
        if map_obj.valid_position(x, y):
            theta = np.random.uniform(0, 2*np.pi)
            return x, y, theta

def initialize_particles(map_obj, N):
    particles = []
    while len(particles) < N:
        x, y, theta = random_valid_pose(map_obj)
        particles.append([x, y, theta])
    return np.array(particles)

def amcl(map_obj, laser, N=1000, step_translation=0.05, alpha=[0.15, 0.15, 0.1, 0.1]):
    robot_x, robot_y, robot_theta = random_valid_pose(map_obj)
    robot_pose_prev = Pose(robot_x, robot_y, robot_theta)
    particles = initialize_particles(map_obj, N)

    fig, ax = plt.subplots(figsize=(15, 15))
    plt.subplots_adjust(bottom=0.15)
    button_ax = plt.axes([0.4, 0.02, 0.2, 0.05])
    kidnap_button = Button(button_ax, 'Kidnap Robot')
    kidnap_triggered = [False]

    def kidnap(event):
        kidnap_triggered[0] = True
        print("Kidnap button clicked!")

    kidnap_button.on_clicked(kidnap)

    wslow = 0.0
    wfast = 0.0
    alpha_slow = 0.01
    alpha_fast = 0.1
    trajectory_x = []
    trajectory_y = []

    while True:
        if kidnap_triggered[0]:
            robot_x, robot_y, robot_theta = random_valid_pose(map_obj)
            robot_pose_prev = Pose(robot_x, robot_y, robot_theta)
            trajectory_x.clear()
            trajectory_y.clear()
            kidnap_triggered[0] = False
            print("Robot kidnapped!")

        ax.clear()
        ax.imshow(map_obj.img, cmap='gray', origin='upper')

        if np.random.rand() < 0.1:
            robot_theta += np.random.uniform(-np.pi/4, np.pi/4)
        else:
            robot_theta += np.radians(10) * (1 if np.random.rand() < 0.5 else -1)
        robot_theta %= 2*np.pi

        new_x = robot_x + step_translation * np.cos(robot_theta)
        new_y = robot_y + step_translation * np.sin(robot_theta)

        if map_obj.valid_position(new_x, new_y):
            robot_pose_curr = Pose(new_x, new_y, robot_theta)
            robot_x, robot_y = new_x, new_y
        else:
            robot_theta += np.pi / 2
            robot_pose_curr = Pose(robot_x, robot_y, robot_theta)

        trajectory_x.append(robot_x)
        trajectory_y.append(robot_y)

        u_t = (robot_pose_prev, robot_pose_curr)
        new_particles = []
        for x, y, theta in particles:
            p = Pose(x, y, theta)
            x2, y2, t2 = sample_motion_model_odometry(u_t, p, alpha)
            if map_obj.valid_position(x2, y2):
                new_particles.append([x2, y2, t2])
            else:
                new_particles.append([x, y, theta])
        particles = np.array(new_particles)
        robot_pose_prev = robot_pose_curr

        real_reading = laser.raycasting(robot_x, robot_y, robot_theta)

        weights = []
        for x, y, theta in particles:
            reading = laser.raycasting(x, y, theta)
            weight = beam_range_finder_model(real_reading, reading, laser.laser_max_range)
            weights.append(weight)
        weights = np.array(weights)
        weights += 1e-300
        weights /= np.sum(weights)

        wavg = np.mean(weights)
        if wslow == 0.0: wslow = wavg
        if wfast == 0.0: wfast = wavg
        wslow += alpha_slow * (wavg - wslow)
        wfast += alpha_fast * (wavg - wfast)
        """
        estimation_error = np.linalg.norm([np.average(particles[:,0], weights=weights) - robot_x,
                                           np.average(particles[:,1], weights=weights) - robot_y])
        if estimation_error > 1.0:
            new_particles = []
            for i in range(N):
                if i < N // 2:
                    idx = np.random.choice(len(particles), p=weights)
                    new_particles.append(particles[idx])
                else:
                    x_rand, y_rand, theta_rand = random_valid_pose(map_obj)
                    new_particles.append([x_rand, y_rand, theta_rand])
            particles = np.array(new_particles)
        else:
            new_particles = []
            for _ in range(N):
                prob = max(0.0, 1.0 - wfast / wslow)
                if np.random.rand() < prob:
                    x_rand, y_rand, theta_rand = random_valid_pose(map_obj)
                    new_particles.append([x_rand, y_rand, theta_rand])
                else:
                    idx = np.random.choice(len(particles), p=weights)
                    new_particles.append(particles[idx])
            particles = np.array(new_particles)

        particles[:, 0] += np.random.normal(0, 0.01, size=N)
        particles[:, 1] += np.random.normal(0, 0.01, size=N)
        particles[:, 2] += np.random.normal(0, np.radians(2), size=N)
        """
        new_particles = []
        new_weights = []
        wavg = 0.0
        for i in range(N):
            x, y, theta = particles[i]
            p = Pose(x, y, theta)
            x2, y2, t2 = sample_motion_model_odometry(u_t, p, alpha)
            if map_obj.valid_position(x2, y2):
                x_new, y_new, theta_new = x2, y2, t2
            else:
                x_new, y_new, theta_new = x, y, theta
            reading = laser.raycasting(x_new, y_new, theta_new)
            weight = beam_range_finder_model(real_reading, reading, laser.laser_max_range)
            new_particles.append([x_new, y_new, theta_new])
            new_weights.append(weight)
            wavg += weight / N  # incremental average

        particles = np.array(new_particles)
        weights = np.array(new_weights)
        #weights += 1e-300
        weights /= np.sum(weights)

        # 2. Atualização dos parâmetros adaptativos
        if wslow == 0.0: wslow = wavg
        if wfast == 0.0: wfast = wavg
        wslow += alpha_slow * (wavg - wslow)
        wfast += alpha_fast * (wavg - wfast)

        # 3. Ressampling adaptativo (Augmented MCL)
        new_particles = []
        prob_random = max(0.0, 1.0 - wfast / wslow)
        for _ in range(N):
            if np.random.rand() < prob_random:
                # Adiciona partícula aleatória
                while True:
                    x_rand = np.random.uniform(0, map_obj.width * map_obj.resolution)
                    y_rand = np.random.uniform(0, map_obj.height * map_obj.resolution)
                    if map_obj.valid_position(x_rand, y_rand):
                        theta_rand = np.random.uniform(0, 2*np.pi)
                        new_particles.append([x_rand, y_rand, theta_rand])
                        break
            else:
                idx = np.random.choice(len(particles), p=weights)
                new_particles.append(particles[idx])
        particles = np.array(new_particles)

        # 4. Adiciona ruído
        particles[:, 0] += np.random.normal(0, 0.01, size=N)
        particles[:, 1] += np.random.normal(0, 0.01, size=N)
        particles[:, 2] += np.random.normal(0, np.radians(2), size=N)
        #x_est = np.average(particles[:, 0], weights=weights)
        #y_est = np.average(particles[:, 1], weights=weights)
        #sin_sum = np.average(np.sin(particles[:, 2]), weights=weights)
        #cos_sum = np.average(np.cos(particles[:, 2]), weights=weights)
        #theta_est = np.arctan2(sin_sum, cos_sum)

        px, py = map_obj.pixel_position(particles[:, 0], particles[:, 1])
        ax.scatter(px, py, s=2, c='orange', label='Particles')
        rx, ry = map_obj.pixel_position(np.array([robot_x]), np.array([robot_y]))
        ax.scatter(rx, ry, s=80, facecolors='none', edgecolors='blue', linewidths=2, label='Robot')
        tx, ty = map_obj.pixel_position(np.array(trajectory_x), np.array(trajectory_y))
        ax.plot(tx, ty, color='cyan', linewidth=1.5, label='Trajectory')

        angles = np.radians(np.arange(0, 360, 3))
        phi = robot_theta + angles
        ranges = real_reading
        x_laser = robot_x + ranges * np.cos(phi)
        y_laser = robot_y + ranges * np.sin(phi)
        lx, ly = map_obj.pixel_position(x_laser, y_laser)
        ax.scatter(lx, ly, s=2, c='red', label='Laser')

        ax.legend()
        ax.set_title("Monte Carlo Localization with Kidnap Button")
        plt.pause(0.05)

def MonteCarloLocalization():
    map_obj, laser = initialize_map_and_laser()
    amcl(map_obj, laser)

if __name__ == "__main__":
    MonteCarloLocalization()
