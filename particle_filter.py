import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, num_landmarks, img_shape, noise_std=5):
        self.num_particles = num_particles
        self.num_landmarks = num_landmarks
        self.img_shape = img_shape
        self.noise_std = noise_std

        # Initialize particles randomly within the image bounds
        self.particles = np.random.rand(num_particles, num_landmarks, 2) * np.array(img_shape[::-1])
        self.weights = np.ones((num_particles, num_landmarks)) / num_particles

    def predict(self):
        # Add random noise to particles (motion model)
        noise = np.random.normal(0, self.noise_std, self.particles.shape)
        self.particles += noise
        # Ensure particles stay within image bounds
        self.particles = np.clip(self.particles, 0, np.array(self.img_shape[::-1]))

    def update(self, observed_landmarks):
        # Update weights based on observed landmarks
        for l in range(self.num_landmarks):
            distances = np.linalg.norm(self.particles[:, l, :] - observed_landmarks[l], axis=1)
            self.weights[:, l] = np.exp(-distances**2 / (2 * self.noise_std**2))
            self.weights[:, l] /= np.sum(self.weights[:, l])  # Normalize weights

    def resample(self):
        # Resample particles based on weights
        for l in range(self.num_landmarks):
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights[:, l])
            self.particles[:, l, :] = self.particles[indices, l, :]
            self.weights[:, l] = 1.0 / self.num_particles  # Reset weights

    def estimate(self):
        # Compute the weighted average of particles for each landmark
        return np.sum(self.particles * self.weights[:, :, np.newaxis], axis=0)