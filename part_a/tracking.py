import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from common.plotting import savefig

def make_tracks(seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 50)
    x1 = 10 + 1.5*t + rng.normal(0, 1.35, len(t))
    y1 = 15 + 0.7*t + rng.normal(0, 1.35, len(t))
    x2 = 90 - 1.2*t + rng.normal(0, 1.35, len(t))
    y2 = 20 + 25*np.sin(t/18) + rng.normal(0, 1.35, len(t))
    pts1 = np.c_[x1, y1, t, np.zeros_like(t)]
    pts2 = np.c_[x2, y2, t, np.ones_like(t)]
    clutter = np.c_[rng.uniform(0,100,160), rng.uniform(0,100,160), rng.uniform(0,50,160), -np.ones(160)]
    return np.vstack([pts1, pts2, clutter])

def mht_like_filter(points):
    # Lightweight reproduction of MHT idea: retain temporally coherent dense supports.
    # We cluster in (x,y,t) after scaling time to separate coherent trajectories from clutter.
    X = points[:, :3].copy()
    X[:, 2] *= 1.7
    labels = DBSCAN(eps=6.0, min_samples=5).fit_predict(X)
    good = labels >= 0
    return points[good], labels[good]

def reconstruct_tracks(cleaned, labels):
    tracks = []
    for lab in sorted(set(labels)):
        pts = cleaned[labels == lab]
        if len(pts) < 8:
            continue
        pts = pts[np.argsort(pts[:,2])]
        tracks.append(pts)
    return tracks

def mean_deviation_to_true(tracks):
    # Synthetic metric: distance from track points to their generating class centerline.
    errs = []
    for tr in tracks:
        for x,y,t,label in tr:
            if label == 0:
                xt, yt = 10 + 1.5*t, 15 + 0.7*t
            elif label == 1:
                xt, yt = 90 - 1.2*t, 20 + 25*np.sin(t/18)
            else:
                continue
            errs.append(np.hypot(x-xt, y-yt))
    return float(np.mean(errs)) if errs else None

def demo_tracking():
    pts = make_tracks()
    cleaned, labels = mht_like_filter(pts)
    tracks = reconstruct_tracks(cleaned, labels)
    mean_dev = mean_deviation_to_true(tracks)

    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=8, alpha=0.45)
    ax.set_title("Raw spatio-temporal detections")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("t")

    ax2 = fig.add_subplot(122, projection="3d")
    for tr in tracks:
        ax2.plot(tr[:,0], tr[:,1], tr[:,2], lw=2)
        ax2.scatter(tr[:,0], tr[:,1], tr[:,2], s=8)
    ax2.set_title("Clutter-suppressed reconstructed tracks")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("t")
    p = savefig("part_a_05_mht_tracking.png")

    return {
        "raw_points": int(len(pts)),
        "cleaned_points": int(len(cleaned)),
        "clutter_removed": int(len(pts)-len(cleaned)),
        "num_tracks": len(tracks),
        "mean_deviation_units": mean_dev,
        "figure": str(p)
    }

if __name__ == "__main__":
    print(demo_tracking())
