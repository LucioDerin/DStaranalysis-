from matplotlib import pyplot as plt
import numpy as np
import sys

def plot_log(epochs, train_loss, train_particles_loss, train_jets_loss, val_loss, val_particles_loss, val_jets_loss, lr, fname):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.5)
    
    # Main plot
    ax1 = fig.add_subplot(gs[:, 0])
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training/Val Loss', color=color)
    ax1.plot(epochs, train_loss, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.plot(epochs, val_loss, color='tab:green', label='Validation Loss')
    ax1.scatter(np.argmin(val_loss), min(val_loss), color='tab:red', marker='*', label=f'Best Val Loss\nEpoch {epochs[np.argmin(val_loss)]}')
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Learning Rate', color=color)  
    ax2.plot(epochs, lr, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_title('Training/Validation Loss and Learning Rate over Epochs')

    # Right subplots (empty placeholders)
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    # Plot train val particles and jets loss
    ax3.plot(epochs, 0.3*np.array(train_particles_loss), label='Train Particles Loss', color='tab:blue')
    ax3.plot(epochs, 0.3*np.array(val_particles_loss), label='Val Particles Loss', color='tab:orange')
    ax3.scatter(np.argmin(val_loss), 0.3*val_particles_loss[np.argmin(val_loss)], color='tab:red', marker='*', label=f'Selected model')
    ax3.set_title('Train/ Val Particles Loss')
    ax3.legend()

    ax4.plot(epochs, train_jets_loss, label='Train Jets Loss', color='tab:blue')
    ax4.plot(epochs, val_jets_loss, label='Val Jets Loss', color='tab:orange')
    ax4.scatter(np.argmin(val_loss), val_jets_loss[np.argmin(val_loss)], color='tab:red', marker='*', label=f'Selected model')
    ax4.set_title('Train/ Val Jets Loss')
    ax4.legend()

    plt.savefig(fname, dpi=300)

if __name__ == "__main__":

    log = sys.argv[1]
    es = []
    tls = []
    tjls = []
    tpls = []
    vls = []
    vjls = []
    vpjs = []
    lrs = []
    with open(log, 'r') as f:
        for line in f:
            line = line.split(",")
            es.append(int( line[0].replace("Epoch ", "").strip().split("/")[0] ))

            tls.append(float(line[1].replace(" Train Loss: ", "").strip()))
            tpls.append(float(line[2].replace("Train Particles Loss:","").strip()))
            tjls.append(float(line[3].replace("Train Jets Loss:","").strip()))

            vls.append(float(line[4].replace(" Val Loss: ", "").strip()))
            vpjs.append(float(line[5].replace("Val Particles Loss:","").strip()))
            vjls.append(float(line[6].replace("Val Jets Loss:","").strip()))

            lrs.append(float(line[7].replace(" LR: ", "").strip()))

    plot_log(es, tls, tpls, tjls, vls, vpjs, vjls, lrs,  fname=f'{log.split(".")[0]}.png')