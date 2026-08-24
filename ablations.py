#%% 
import matplotlib.pyplot as plt
import numpy as np
def plot_lines(x_values, data_vectors, labels):
    # Define colors and line styles
    colors = ['b', 'r', 'g','k']#, 'm', 'y', 'k']
    line_styles = ['-', '--', '-.', ':']#, '.', 'o', 'v']

    # Assume data_vectors is a list of 1D arrays or lists, and labels is a list of strings
    for i, (data, label) in enumerate(zip(data_vectors, labels)):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        plt.plot(x_values, data, label=label, color=color, linestyle=line_style)

        # Denote the maximum value with a triangle
        max_index = np.argmax(data)
        plt.plot(x_values[max_index], data[max_index], marker='*', color='k')

    plt.xlabel('SSL Augmentation Probability')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%
#Labeled Data Size - Arousal Valence
x_values = [6.9,11.6,16.3,20.9,25.6,30.2,34.9,39.5]
data_vectors = [[0.544,0.519,0.573,0.537,0.574,0.556,0.532,0.549], [0.531,0.526,0.524,0.554,0.518,0.515,0.521,0.572]]
labels = ['Arousal', 'Valence']
plot_lines(x_values,data_vectors, labels)

# %%
#SSL task probability - Valence
x_values = [0.05,0.1,0.15,0.20,0.25]  # Replace with your actual x values
data_vectors = [
    [0.535,0.571,0.544,0.565,0.586],
    [0.556,0.568,0.589,0.552,0.546],
    [0.539,0.534,0.536,0.591,0.558],
    [0.574,0.568,0.562,0.552,0.537]
]
labels = ['Node Attribute Noise', 'Node Attribute Masking', 'Node Masking','Edge Removal']
plot_lines(x_values, data_vectors, labels)

#%%
#SSL task probability - Arousal
x_values = [0.05,0.1,0.15,0.20,0.25]  # Replace with your actual x values
data_vectors = [
    [0.54,0.548,0.559,0.552,0.541],
    [0.578,0.534,0.540,0.573,0.581],
    [0.558,0.575,0.546,0.549,0.568 ],
    [0.556,0.524,0.563,0.54,0.552]
]
labels = ['Node Attribute Noise', 'Node Attribute Masking', 'Node Masking','Edge Removal']
plot_lines(x_values, data_vectors, labels)

# %%
