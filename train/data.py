import argparse
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb


# Default parameters of the "class_ring_mixture" task (overridden key by key by task_params / config['task']['ring']).
# Anchors A, B, C, D on the unit circle; boundary i is between anchor i and anchor i+1 (cyclic): A|B, B|C, C|D, D|A.
RING_DEFAULTS = {
    'radius': 1.0,
    'angles_deg': [0, 90, 180, 270],  # Anchor angles, strictly increasing in [0, 360)
    'sigmas': [0.4, 0.4, 0.4, 0.4],  # Isotropic std of each Gaussian (input density only)
    'weights': [0.25, 0.25, 0.25, 0.25],  # Mixing weights (normalized)
    'label_dists': [  # P(y | anchor i), one row per anchor, num_classes columns, strictly positive
        [0.86, 0.04, 0.05, 0.05],
        [0.20, 0.70, 0.05, 0.05],
        [0.24, 0.24, 0.28, 0.24],
        [0.05, 0.04, 0.05, 0.86],
    ],
    # Transition width of boundary i (between anchor i and i+1) as a fraction of the arc between them, in [0, 1].
    # 0 = hard step at the bisector (nearest-anchor labels); 1 = linear ramp from anchor i to anchor i+1.
    'transition_widths': [0.0, 0.0, 0.0, 1.0],
    'interp_space': 'prob',  # 'prob': (1-w) P_i + w P_j; 'logit': softmax((1-w) ln P_i + w ln P_j)
    'targets': 'sample',  # 'sample': one-hot y ~ P(y|x); 'soft': the full P(y|x)
}

ANCHOR_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class RingMixtureTask:
    """
    2D input density: a mixture of K isotropic Gaussians whose means (anchors) lie on a circle.
    Label function: depends only on the input's angle phi. Between adjacent anchors i and j = i+1 (cyclic), with
    t = (phi - theta_i) / (theta_j - theta_i) in [0, 1] and width tau_i of boundary i,
        w = 1[t > 1/2]                         if tau_i = 0
        w = clip((t - 1/2) / tau_i + 1/2, 0, 1)  otherwise
        P(y|x) = (1 - w) P_i + w P_j             (interp_space 'prob'; 'logit' mixes log-probabilities instead)
    So the input density (sigmas, weights) and the label function (label_dists, transition_widths) are independent.
    """

    def __init__(self, num_classes: int, params: dict = None):
        params = {**RING_DEFAULTS, **(params or {})}
        unknown = set(params) - set(RING_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown ring task parameters: {sorted(unknown)}")
        self.params = params

        self.radius = float(params['radius'])
        self.angles = np.deg2rad(np.asarray(params['angles_deg'], dtype=np.float64))
        self.sigmas = np.asarray(params['sigmas'], dtype=np.float64)
        weights = np.asarray(params['weights'], dtype=np.float64)
        dists = np.asarray(params['label_dists'], dtype=np.float64)
        self.widths = np.asarray(params['transition_widths'], dtype=np.float64)
        self.interp_space = params['interp_space']
        self.target_mode = params['targets']
        K = len(self.angles)
        self.K = K
        self.num_classes = num_classes

        if num_classes <= 2:
            raise ValueError("class_ring_mixture needs num_classes > 2 (CrossEntropyLoss with one output per class)")
        if K < 2:
            raise ValueError("class_ring_mixture needs at least 2 anchors")
        if np.any(self.angles < 0) or np.any(self.angles >= 2 * np.pi) or np.any(np.diff(self.angles) <= 0):
            raise ValueError(f"angles_deg must be strictly increasing in [0, 360), got {params['angles_deg']}")
        for name, arr in [('sigmas', self.sigmas), ('weights', weights), ('transition_widths', self.widths)]:
            if arr.shape != (K,):
                raise ValueError(f"{name} must have one entry per anchor ({K}), got {arr.tolist()}")
        if np.any(self.sigmas <= 0) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("sigmas must be positive and weights non-negative with a positive sum")
        if np.any(self.widths < 0) or np.any(self.widths > 1):
            raise ValueError(f"transition_widths must be in [0, 1], got {self.widths.tolist()}")
        if dists.shape != (K, num_classes):
            raise ValueError(f"label_dists must be [{K} anchors, {num_classes} classes], got shape {list(dists.shape)}")
        if np.any(dists <= 0):
            raise ValueError("label_dists must be strictly positive (KL divergences and surprisals would be infinite)")
        if not np.allclose(dists.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError(f"Each row of label_dists must sum to 1, got sums {dists.sum(axis=1).tolist()}")
        sorted_dists = np.sort(dists, axis=1)
        if np.any(sorted_dists[:, -1] == sorted_dists[:, -2]):
            raise ValueError("Each anchor's label distribution needs a unique max-probability class")
        if len(set(dists.argmax(axis=1).tolist())) != K:
            raise ValueError(f"Anchors need distinct max-probability classes, got {dists.argmax(axis=1).tolist()}")
        if self.interp_space not in ('prob', 'logit'):
            raise ValueError(f"interp_space must be 'prob' or 'logit', got {self.interp_space}")
        if self.target_mode not in ('sample', 'soft'):
            raise ValueError(f"targets must be 'sample' or 'soft', got {self.target_mode}")

        self.weights = weights / weights.sum()
        self.label_dists = dists / dists.sum(axis=1, keepdims=True)
        self.means = self.radius * np.stack([np.cos(self.angles), np.sin(self.angles)], axis=1)  # [K, 2]
        self.names = [ANCHOR_NAMES[i] if i < len(ANCHOR_NAMES) else str(i) for i in range(K)]

    def sample_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw n inputs [n, 2] from the Gaussian mixture."""
        comps = rng.choice(self.K, size=n, p=self.weights)
        return self.means[comps] + self.sigmas[comps, None] * rng.standard_normal((n, 2))

    def density(self, X) -> np.ndarray:
        """Input density of the Gaussian mixture at X [n, 2]. Returns [n]."""
        X = np.asarray(X, dtype=np.float64)
        sq = ((X[:, None, :] - self.means[None]) ** 2).sum(-1)  # [n, K]
        comps = self.weights * np.exp(-sq / (2 * self.sigmas ** 2)) / (2 * np.pi * self.sigmas ** 2)
        return comps.sum(axis=1)

    def _segment(self, X: np.ndarray):
        """Segment index i (between anchors i and i+1) and position t in [0, 1] along it for each input."""
        phi = np.arctan2(X[:, 1], X[:, 0]) % (2 * np.pi)
        # Segment i covers [theta_i, theta_{i+1}); the last one wraps around through 2*pi
        seg = (np.searchsorted(self.angles, phi, side='right') - 1) % self.K
        start = self.angles[seg]
        gap = (np.roll(self.angles, -1) - self.angles) % (2 * np.pi)
        gap[gap == 0] = 2 * np.pi  # Only with a single anchor; kept for safety
        t = ((phi - start) % (2 * np.pi)) / gap[seg]
        return seg, t

    def mix_weight(self, t: np.ndarray, width: np.ndarray) -> np.ndarray:
        """Weight w of the next anchor at position t along a segment with transition width `width`."""
        hard = (t > 0.5).astype(np.float64)
        safe = np.where(width > 0, width, 1.0)
        soft = np.clip((t - 0.5) / safe + 0.5, 0.0, 1.0)
        return np.where(width > 0, soft, hard)

    def interpolate(self, i: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Label distributions between anchor i and anchor i+1 at mixing weights w. Returns [n, num_classes]."""
        j = (i + 1) % self.K
        P_i, P_j = self.label_dists[i], self.label_dists[j]
        w = w[:, None]
        if self.interp_space == 'prob':
            return (1 - w) * P_i + w * P_j
        log_p = (1 - w) * np.log(P_i) + w * np.log(P_j)
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        return p / p.sum(axis=1, keepdims=True)

    def true_probs(self, X) -> np.ndarray:
        """Ground-truth P(y|x) [n, num_classes] for inputs X [n, 2] (numpy array or tensor)."""
        X = np.asarray(X, dtype=np.float64)
        seg, t = self._segment(X)
        return self.interpolate(seg, self.mix_weight(t, self.widths[seg]))

    def targets(self, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Training targets: one-hot samples y ~ P(y|x), or P(y|x) itself."""
        if self.target_mode == 'soft':
            return probs
        cdf = np.cumsum(probs, axis=1)
        u = rng.random((probs.shape[0], 1)) * cdf[:, -1:]
        labels = np.minimum((u > cdf).sum(axis=1), self.num_classes - 1)
        return np.eye(self.num_classes)[labels]

    def boundary_stats(self) -> list:
        """
        Information-theoretic quantities for each boundary between adjacent anchors i and j (nats):
        KL in both directions, surprisal of each anchor's argmax under the other (-ln P_i(argmax P_j) and
        -ln P_j(argmax P_i)), entropies, and where along the arc (t in [0, 1]) the true argmax changes.
        """
        def kl(p, q):
            return float((p * np.log(p / q)).sum())

        def entropy(p):
            return float(-(p * np.log(p)).sum())

        t_grid = np.linspace(0, 1, 20001)
        stats = []
        for i in range(self.K):
            j = (i + 1) % self.K
            P_i, P_j = self.label_dists[i], self.label_dists[j]
            width = np.full_like(t_grid, self.widths[i])
            argmax = self.interpolate(np.full(t_grid.shape, i), self.mix_weight(t_grid, width)).argmax(axis=1)
            flips = np.nonzero(np.diff(argmax))[0]
            stats.append({
                'boundary': f"{self.names[i]}|{self.names[j]}",
                'width': float(self.widths[i]),
                'kl_ij': kl(P_i, P_j),
                'kl_ji': kl(P_j, P_i),
                'surprisal_i_of_j': float(-np.log(P_i[P_j.argmax()])),
                'surprisal_j_of_i': float(-np.log(P_j[P_i.argmax()])),
                'entropy_i': entropy(P_i),
                'entropy_j': entropy(P_j),
                # Positions where the argmax of the true P(y|x) changes (0.5 for a hard boundary)
                'argmax_flips_t': [float(0.5 * (t_grid[k] + t_grid[k + 1])) for k in flips],
            })
        return stats

    def print_stats(self):
        print(f"Ring mixture: {self.K} anchors, {self.num_classes} classes, interp_space={self.interp_space}, targets={self.target_mode}")
        for name, angle, sigma, weight, dist in zip(self.names, np.rad2deg(self.angles), self.sigmas, self.weights, self.label_dists):
            print(f"  {name}: angle {angle:6.1f} deg, sigma {sigma:.3f}, weight {weight:.3f}, P(y) = {np.round(dist, 3).tolist()}")
        print("  i|j   width  KL(i||j)  KL(j||i)  -lnP_i(argmax j)  -lnP_j(argmax i)  H_i    H_j    argmax flips at t")
        for s in self.boundary_stats():
            flips = ", ".join(f"{t:.3f}" for t in s['argmax_flips_t'])
            print(f"  {s['boundary']:<5} {s['width']:<6.2f} {s['kl_ij']:<9.3f} {s['kl_ji']:<9.3f} {s['surprisal_i_of_j']:<17.3f} "
                  f"{s['surprisal_j_of_i']:<17.3f} {s['entropy_i']:<6.3f} {s['entropy_j']:<6.3f} {flips}")


class ToyDataset(Dataset):
    """
    Synthetic dataset for toy regression tasks.
    Generates data on-the-fly (or pre-generated in memory) for consistency.

    task_params: extra task parameters. For "class_ring_mixture", the dict under config['task']['ring']
    (see RING_DEFAULTS); ignored by the other tasks.
    """
    def __init__(self, task_name: str, num_samples: int = 10000, noise_std: float = 0.05, num_classes: int = 3,
                 distribution: str = 'uniform', seed=0, task_params: dict = None):
        self.task_name = task_name
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.num_classes = num_classes
        self.distribution = distribution
        self.seed = seed

        self.ring = None
        if task_name == "class_ring_mixture":
            if noise_std:
                raise ValueError("class_ring_mixture has no input noise (noise_std must be 0); "
                                 "the input spread is set per anchor by ring.sigmas")
            self.ring = RingMixtureTask(num_classes, task_params)
        self.probs = None  # Ground-truth P(y|x) [num_samples, num_classes] (class_ring_mixture only)

        self.shuffle_count = 0

        self.data, self.targets = self._generate_data()

    def shuffle(self):
        """Shuffle the dataset by incrementing the shuffle count."""
        self.shuffle_count += 1
        self.data, self.targets = self._generate_data()

    def _rng(self):
        # reset seed for each shuffle
        return np.random.default_rng(self.seed + self.shuffle_count * 1000)

    def _sample(self):
        rng = self._rng()

        # Generate 2D inputs in range [-1, 1]
        if self.distribution == 'uniform':
            X = rng.uniform(-1, 1, size=(self.num_samples, 2)).astype(np.float32)
        elif self.distribution == 'normal':
            X = rng.normal(0, 1, size=(self.num_samples, 2)).astype(np.float32)
        return X

    def _get_y(self, X):
        if self.task_name == "reg_sine_wave":
            # Task: f(x1, x2) = {sin(x1) + cos(x2)} * INDICATOR(x1 < 0)
            # Hamayun et al. (2024): https://arxiv.org/pdf/2402.15555
            x1 = X[:, 0]
            x2 = X[:, 1]
            indicator = np.where(x1 < 0, 1, 0)
            y = (np.sin(x1) + np.cos(x2)) * indicator

        elif self.task_name == "class_spiral":
            curvature = 4.0

            # Convert to Polar Coordinates relative to center (0,0)
            r = np.linalg.norm(X, axis=1)
            theta = np.arctan2(X[:, 1], X[:, 0])

            # Points further out are rotated more, creating the spiral effect.
            theta_twisted = theta + (curvature * r)

            # Rescale angles to [0, 2pi]
            theta_twisted = theta_twisted % (2 * np.pi)

            # Assign Classes based on Angle Sectors
            y = np.floor((theta_twisted / (2 * np.pi)) * self.num_classes).reshape(-1, 1).astype(int)

            # if num_classes == 2, we don't need to one-hot encode the targets
            if self.num_classes > 2:
                # One-hot encode the targets
                y = np.eye(self.num_classes)[y.squeeze(1)]

        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        return y.astype(np.float32)

    def _generate_ring(self):
        # Inputs, ground-truth P(y|x) and sampled labels all come from the seeded generator; the input distribution
        # setting and the post-hoc input noise of the other tasks do not apply.
        rng = self._rng()
        X = self.ring.sample_inputs(rng, self.num_samples)
        probs = self.ring.true_probs(X)
        y = self.ring.targets(probs, rng)
        self.probs = torch.from_numpy(probs.astype(np.float32))
        return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

    def _generate_data(self):
        if self.ring is not None:
            return self._generate_ring()

        X = self._sample()
        y = self._get_y(X)

        # Add some small Gaussian noise to the inputs
        X += np.random.normal(0, self.noise_std, size=X.shape)

        return torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def visualize_dataset():
    print("Generating Spiral Dataset...")
    num_classes = 2
    noise_std=0.00
    dataset = ToyDataset(task_name="class_spiral", num_samples=5000, num_classes=num_classes, noise_std=noise_std, seed=42)
    X = dataset.data.numpy()
    y = dataset.targets.numpy()

    # Create the plot
    plt.figure(figsize=(8, 8), dpi=300)

    # Scatter plot: Color points by class label
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', s=5, alpha=0.6)

    # Aesthetics
    # plt.colorbar(scatter, label="Class Label")
    plt.title(f"Twisted Spiral Task ({num_classes} Classes)\nManifold for ResNet Activation Analysis")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # # Show plot
    # print("Displaying plot...")
    # plt.show()

    # Save the plot
    plt.savefig(f"plots/spiral_dataset_{num_classes}classes_{noise_std}std.png")
    print(f"Plot saved as 'plots/spiral_dataset_{num_classes}classes_{noise_std}std.png'.")


# Categorical class colors (fixed order); classes beyond the 8th share a muted gray
CLASS_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
MUTED = '#898781'


def visualize_ring_dataset(num_classes: int, task_params: dict = None, num_samples: int = 5000, seed: int = 42,
                           lim: float = 2.0, save_path: str = "plots/ring_mixture_dataset.png"):
    """Samples colored by their label, and the ground-truth argmax / entropy of P(y|x) over the input plane."""
    dataset = ToyDataset("class_ring_mixture", num_samples=num_samples, noise_std=0.0, num_classes=num_classes,
                         seed=seed, task_params=task_params)
    ring = dataset.ring
    ring.print_stats()
    X = dataset.data.numpy()
    labels = dataset.targets.numpy().argmax(axis=1)
    colors = [CLASS_COLORS[c] if c < len(CLASS_COLORS) else MUTED for c in range(num_classes)]

    axis = np.linspace(-lim, lim, 400)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    probs = ring.true_probs(grid)
    argmax = probs.argmax(axis=1).reshape(len(axis), len(axis))
    entropy = (-(probs * np.log(probs)).sum(axis=1)).reshape(len(axis), len(axis))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    extent = (-lim, lim, -lim, lim)

    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=[colors[c] for c in labels], s=3, alpha=0.6, linewidths=0)
    ax.set_title(f"Samples (targets: {ring.target_mode})")

    ax = axes[1]
    # Argmax class, faded where the true distribution is flat (max probability near 1/num_classes)
    max_p = probs.max(axis=1).reshape(len(axis), len(axis))
    alpha = np.clip((max_p - 1 / num_classes) / (1 - 1 / num_classes), 0, 1) * 0.8 + 0.2
    rgba = np.array([to_rgb(c) for c in colors])[argmax]
    ax.imshow(np.concatenate([rgba, alpha[..., None]], axis=-1), origin='lower', extent=extent)
    ax.set_title("True argmax P(y|x) (opacity: max probability)")

    ax = axes[2]
    im = ax.imshow(entropy, origin='lower', extent=extent, cmap='Blues', vmin=0, vmax=np.log(num_classes))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Entropy of P(y|x) [nats]")
    ax.set_title("True entropy")

    for ax in axes:
        ax.scatter(ring.means[:, 0], ring.means[:, 1], c='black', s=30, marker='x')
        for name, (mx, my) in zip(ring.names, ring.means):
            ax.annotate(name, (mx, my), xytext=(6, 6), textcoords='offset points', fontsize=12)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    handles = [plt.Line2D([], [], marker='o', linestyle='', color=colors[c], label=f"class {c}") for c in range(min(num_classes, len(CLASS_COLORS)))]
    if num_classes > len(CLASS_COLORS):
        handles.append(plt.Line2D([], [], marker='o', linestyle='', color=MUTED, label="other classes"))
    axes[0].legend(handles=handles, loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Plot saved as '{save_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a toy dataset")
    parser.add_argument("--task", type=str, default="class_spiral", choices=["class_spiral", "class_ring_mixture"])
    parser.add_argument("--config", type=str, default="train/config.yaml",
                        help="Training config whose task.num_classes and task.ring are used for class_ring_mixture")
    parser.add_argument("--save_path", type=str, default="plots/ring_mixture_dataset.png")
    args = parser.parse_args()
    if args.task == "class_ring_mixture":
        import yaml
        with open(args.config) as f:
            task_config = yaml.safe_load(f)['task']
        visualize_ring_dataset(task_config['num_classes'], task_config.get('ring'), save_path=args.save_path)
    else:
        visualize_dataset()
