import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(reward,mean_rewards):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Rewards')
    plt.plot(reward)
    plt.plot(mean_rewards)
    plt.ylim(ymin=-25)
    plt.text(len(reward)-1, reward[-1], str(reward[-1]))
    plt.show(block=False)
    plt.pause(.1)
