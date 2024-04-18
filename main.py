import gradio as gr
import librosa
import numpy as np
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import io

def ensure_length(audio, n_fft):
    """Ensure audio is at least n_fft in length by padding with zeros if necessary."""
    if len(audio) < n_fft:
        pad_length = n_fft - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    return audio

def compute_hpcp(audio, sr, hop_length, n_chroma, bins_per_octave):
    """
    Compute HPCP features from an audio signal with higher resolution.
    """
    X = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length, n_chroma=n_chroma, bins_per_octave=bins_per_octave)
    return X.T

def compute_ssm(X):
    """
    Compute the segment similarity matrix (SSM) from an HPCP time series using Librosa's optimized functions.
    """
    # Use librosa's recurrence_matrix for optimized SSM computation
    S = librosa.segment.recurrence_matrix(X, mode='affinity', metric='cosine', sparse=False)
    
    # No need to convert S to dense format, as it's already a numpy array
    return S

def generate_ssm(audio_data, hop_length_factor, n_chroma, bins_per_octave_multiplier, hop_length_multiplier, color_map, threshold):
    sr, audio = audio_data
    # Ensure audio is mono for processing
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    audio = ensure_length(audio, 2048)  # Ensure audio length for n_fft=2048
    
    # Adjust hop_length based on the factor provided by the user
    hop_length = int(0.0695 * sr * hop_length_factor * hop_length_multiplier)
    
    # Calculate bins_per_octave based on the multiplier and n_chroma
    bins_per_octave = int(bins_per_octave_multiplier * n_chroma)
    
    X = compute_hpcp(audio, sr, hop_length, n_chroma, bins_per_octave)
    S = compute_ssm(X)
    S_upscaled = np.kron(S, np.ones((20, 20)))  # Linear upscaling by a factor of 2
    
    # Apply threshold
    S_upscaled[S_upscaled < threshold] = 0
    
    # Apply colormap to the image
    plt.imshow(S_upscaled, cmap=color_map)
    plt.axis('off')
    
    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    
    # Read the image from the buffer using OpenCV
    ssm_image = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), -1)
    buffer.close()  # Close the buffer
    
    return ssm_image

# Define sliders for the parameters
hop_length_factor_slider = gr.Slider(minimum=0.5, maximum=10.0, step=0.1, value=1.0, label="Hop Length Factor")
n_chroma_slider = gr.Slider(minimum=6, maximum=72, step=1, value=24, label="Number of Chroma Bins")  # Adjusted range
bins_per_octave_multiplier_slider = gr.Slider(minimum=1, maximum=10, step=0.1, value=2.0, label="Bins Per Octave Multiplier")  # Adjusted range
hop_length_multiplier_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Hop Length Multiplier")
threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Threshold")

# Define the color map dropdown
color_map_dropdown = gr.Dropdown(choices=["inferno"], label="Color Map")

iface = gr.Interface(
    fn=generate_ssm,
    inputs=[
        gr.Audio(type="numpy", label="Upload Audio"),
        hop_length_factor_slider,
        n_chroma_slider,
        bins_per_octave_multiplier_slider,
        hop_length_multiplier_slider,
        color_map_dropdown,
        threshold_slider
    ],
    outputs=gr.Image(label="SSM Map"),
    title="Music Structure Analysis",
    description="Upload an audio file to generate its segment similarity matrix (SSM). Adjust the parameters to change the resolution of the SSM and the number of subdivisions. Set a threshold for on/off."
)

iface.launch(server_name="0.0.0.0")
