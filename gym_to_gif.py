from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import sys
import numpy as np

"""
Ensure you have imagemagick installed with
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def main():
    filename = sys.argv[1]
    gif_filename = sys.argv[2]
    frames = np.load(filename)
    print(frames.shape)
    save_frames_as_gif(frames, path="results/", filename=gif_filename)

if __name__ == '__main__':
    main()
