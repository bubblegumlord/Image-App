import numpy as np

def get_mask(key):
    masks = {
        # ŚREDNIE
        "mean3": np.ones((3, 3), dtype=np.float32) / 9,
        "mean5": np.ones((5, 5), dtype=np.float32) / 25,

        # WAGOWA (binominalna)
        "weight3": (1 / 16) * np.array([
            [1, 2, 1],
            [2, 5, 2],
            [1, 2, 1]
        ], dtype=np.float32),

        # GAUSS 3x3
        "gauss3": (1 / 16) * np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32),

        # LAPLACE
        "laplace1": np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32),

        "laplace2": np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=np.float32),

        "laplace3": np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ], dtype=np.float32),

        # PREWITT (8 kierunków)
        "prewitt_n": np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.float32),

        "prewitt_s": np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ], dtype=np.float32),

        "prewitt_e": np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32),

        "prewitt_w": np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ], dtype=np.float32),

        "prewitt_ne": np.array([
            [0, -1, -1],
            [1, 0, -1],
            [1, 1, 0]
        ], dtype=np.float32),

        "prewitt_nw": np.array([
            [-1, -1, 0],
            [-1, 0, 1],
            [0, 1, 1]
        ], dtype=np.float32),

        "prewitt_se": np.array([
            [0, 1, 1],
            [-1, 0, 1],
            [-1, -1, 0]
        ], dtype=np.float32),

        "prewitt_sw": np.array([
            [1, 1, 0],
            [1, 0, -1],
            [0, -1, -1]
        ], dtype=np.float32),

        # SOBEL
        "sobel_x": np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32),

        "sobel_y": np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32)
    }

    return masks[key]