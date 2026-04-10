import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(12,5.5))

# timeline
x = np.arange(0,10)

# meeting blocks
train1 = (0,4)
val1   = (4,5)
test1  = (5,6)

train2 = (0,6)
val2   = (6,7)
test2  = (7,8)

train3 = (0,8)
val3   = (8,9)
test3  = (9,10)

blocks = [
    (train1, "Train", "#4CAF50"),
    (val1, "Validate", "#FFC107"),
    (test1, "Test", "#F44336"),

    (train2, "Train", "#4CAF50"),
    (val2, "Validate", "#FFC107"),
    (test2, "Test", "#F44336"),

    (train3, "Train", "#4CAF50"),
    (val3, "Validate", "#FFC107"),
    (test3, "Test", "#F44336"),
]

y_levels = [2,1,0]

i = 0
for (start,end),label,color in blocks:
    y = y_levels[i//3]
    rect = patches.Rectangle((start,y), end-start, 0.6,
                             facecolor=color, edgecolor="black", alpha=0.8)
    ax.add_patch(rect)

    ax.text((start+end)/2, y+0.3, label,
            ha="center", va="center", fontsize=9)

    i += 1

# FOMC meeting markers
meeting_positions = [4,6,8]
for m in meeting_positions:
    ax.axvline(m, linestyle="--", color="black", alpha=0.6)
    ax.text(m,2.7,"FOMC", rotation=90, ha="center", va="bottom")

# labels
ax.set_xlim(0,10)
ax.set_ylim(-0.5,3)

ax.set_yticks([2.3,1.3,0.3])
ax.set_yticklabels([
    "Cycle 1\n(expanding train)",
    "Cycle 2",
    "Cycle 3"
])

ax.set_xlabel("Time (meeting cycles)")
ax.set_title("Planned Backtesting Procedure - Statistical-Arb strategies")

# hyperparameter grid illustration
text = """
Local Parameter Grid

OU window:
  30, 60, 90

Entry z:
  1.5, 2.0, 2.5

Exit z:
  0.0, 0.5

No-trade buffer:
  1–3 days
"""

# place the parameter grid text in the top-right of the figure
fig.text(
    0.98,
    0.98,
    text,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(boxstyle="round", fc="white", ec="black"),
)

plt.tight_layout()
plt.show()