import os
import matplotlib.pyplot as plt

def plot_loss(history, filename="loss_plot.png"):
    # Create the 'plots' folder if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Save the plot to the 'plots' folder
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")