import imageio
import os
import progressbar
import shutil
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted


def visualize_matrix(M, epoch=1):
    fig = plt.figure(figsize=(5,5))
    ax1 = plt.subplot(1,1,1)
    
    title = f"Epoch {epoch}"
    fig.suptitle(title, fontsize=20)
    
    height, width = M.shape
    Mud = np.round(np.flipud(M), 2)
    coordinates = [(i, j) for i in range(height) for j in range(width)]
    for coordinate in coordinates:
        i, j = coordinate
        value = Mud[i, j]
        rel_coordinate = (j/float(width), i/float(height))
        ax1.annotate(value, rel_coordinate, ha='left', va='center',
                     size=22, alpha=0.7, family='serif')
        
    padding = 0.2
    wmargin = (width-1)/float(width) + padding
    hmargin = (height-1)/float(height) + padding
    
    hcenter = np.median(range(height))/float(height)
    hcenter = hcenter + 0.015
    
    bracket_d = 0.4
    bracket_b = 0.05
    bracket_paddingl = 0.05
    bracket_paddingr = -0.05
    
    ax1.plot([-bracket_paddingl, -bracket_paddingl],[hcenter-bracket_d, hcenter+bracket_d], 'k-', lw=2, alpha=0.7)
    ax1.plot([-bracket_paddingl, -bracket_paddingl+bracket_b], [hcenter-bracket_d, hcenter-bracket_d], 'k-', lw=2, alpha=0.7)
    ax1.plot([-bracket_paddingl, -bracket_paddingl+bracket_b], [hcenter+bracket_d, hcenter+bracket_d], 'k-', lw=2, alpha=0.7)
    
    ax1.plot([wmargin-bracket_paddingr, wmargin-bracket_paddingr],[hcenter-bracket_d, hcenter+bracket_d], 'k-', lw=2, alpha=0.7)
    ax1.plot([wmargin-bracket_paddingr-bracket_b, wmargin-bracket_paddingr], [hcenter-bracket_d, hcenter-bracket_d], 'k-', lw=2, alpha=0.7)
    ax1.plot([wmargin-bracket_paddingr-bracket_b, wmargin-bracket_paddingr], [hcenter+bracket_d, hcenter+bracket_d], 'k-', lw=2, alpha=0.7)
    
    ax1.set_xlim([-padding, wmargin + 0.2])
    ax1.set_ylim([-padding, hmargin])
    
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.axis('off')
    
    plt.tight_layout()
    return fig


def create_matrix_gif(values, save_folder="gif_images/", out_image="training_weights.gif"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    out_image_folder = '/'.join(out_image.split('/')[:-1])
    if not os.path.exists(out_image_folder):
        os.makedirs(out_image_folder)
    
    with progressbar.ProgressBar(max_value=len(values)) as bar:
        for i, value in enumerate(values):
            save_path = f"{save_folder}img{str(i)}.png"
            fig = visualize_matrix(value, i+1)
            fig.savefig(save_path)
            plt.close(fig)
            bar.update(i)

    imagepaths = [os.path.join(save_folder, fname) for fname in os.listdir(save_folder) if fname.endswith('.png')]
    imagepaths = natsorted(imagepaths)

    fps = len(imagepaths)//5 if len(imagepaths) < 250 else 50
    with imageio.get_writer(out_image, mode='I', fps=fps) as writer:
        for impath in imagepaths:
            image = imageio.imread(impath)
            writer.append_data(image)
            
    shutil.rmtree(save_folder, ignore_errors=True) # Remove frames for gif
