import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

import matplotlib.patches as patches

#source: https://www.kaggle.com/code/mmoore23/nfl-2024-tracking-data-animations
def create_football_field(
    linenumbers=True,
    endzones=True,
    figsize=(12, 6.33),
    line_color="black",
    field_color="white",
    ez_color=None,
    ax=None,
    return_fig=False,
):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """

    if ez_color is None:
        ez_color = field_color

    rect = patches.Rectangle(
        (0, 0),
        120,
        53.3,
        linewidth=0.1,
        edgecolor="r",
        facecolor=field_color,
        zorder=0,
    )

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)
    ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color=line_color)
    
    # Endzones
    if endzones:
        ez1 = patches.Rectangle(
            (0, 0),
            10,
            53.3,
            linewidth=0.1,
            edgecolor=line_color,
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ez2 = patches.Rectangle(
            (110, 0),
            10,
            53.3,
            linewidth=0.1,
            edgecolor=line_color,
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ax.add_patch(ez1)
        ax.add_patch(ez2)

    ax.axis("off")
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            ax.text(
                x,
                5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color=line_color,
            )
            ax.text(
                x - 0.95,
                53.3 - 5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color=line_color,
                rotation=180,
            )
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color=line_color)
        ax.plot([x, x], [53.0, 52.5], color=line_color)
        ax.plot([x, x], [22.91, 23.57], color=line_color)
        ax.plot([x, x], [29.73, 30.39], color=line_color)

    border = patches.Rectangle(
        (-5, -5),
        120 + 10,
        53.3 + 10,
        linewidth=0.1,
        edgecolor="orange",
        facecolor=line_color,
        alpha=0,
        zorder=0,
    )
    ax.add_patch(border)
    ax.set_xlim((-5, 125))
    ax.set_ylim((-5, 53.3 + 5))

    if return_fig:
        return fig, ax
    else:
        return ax
    

def animate_tracking_data(tracking_df, game_play, movement_players, first_move_frames):
    n = tracking_df[(tracking_df.gameplayid == game_play) ].frameId.max()

    # Initialize the football field plot
    fig, ax = create_football_field(return_fig=True)

    # Get unique club names and assign colors
    clubs = tracking_df[(tracking_df.gameplayid == game_play) & (tracking_df['club'] != 'football')]['club'].unique()
    print(clubs)
    club_colors = {clubs[0]: 'orange', clubs[1]: 'lightblue', 'football': 'brown'}
    
    texts = []  # To store jersey number text elements

    # Initialize the scatter plot for each club.
    scatters = {}
    for club in tracking_df.club.unique():
        color = club_colors.get(club, 'white')
        if club == "football":
            scatters[club] = ax.scatter([], [], label=club, s=80, color=color, lw=1, edgecolors="black", zorder=5)
        else:
            scatters[club] = ax.scatter([], [], label=club, s=170, color=color, lw=1, edgecolors="black", zorder=5)
            
    ax.legend().remove()

    def update(frame):
        # Clear previous frame's texts
        for text in texts:
            text.remove()
        texts.clear()
        

        frame_data = tracking_df[(tracking_df.gameplayid == game_play) & (tracking_df.frameId == frame)]
        event_for_frame = frame_data['event'].iloc[0]  # Assuming each frame has consistent event data
        frame_type=frame_data['frameType'].iloc[0]
        motion_players=0
        for p, f in zip(movement_players, first_move_frames):
            if frame_data['frameId'].iloc[0]==f:
                motion_players+=10
            else:
                continue
        ax.set_title(f"Players in Motion Since Lineset: {motion_players}", fontsize=15)
        for club, d in frame_data.groupby("club"):
            scatters[club].set_offsets(np.c_[d["x"].values, d["y"].values])
            scatters[club].set_color(club_colors.get(club, 'white'))
            scatters[club].set_edgecolors("black")  # Explicitly setting the edge color
            
            # Display jersey numbers if it's not the football
            if club != "football":
                for _, row in d.iterrows():
                    text = ax.text(row["x"], row["y"], str(int(row["jerseyNumber"])), 
                                   fontsize=8, ha='center', va='center', color="black", fontweight='bold', zorder=6)
                    texts.append(text)

    ani = FuncAnimation(fig, update, frames=range(1, n+1), repeat=True, interval=200)
    plt.close(ani._fig)

    # Display the animation in the notebook
    return HTML(ani.to_jshtml())
