import gradio as gr
import librosa
import numpy as np
from scipy.ndimage import convolve

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

def generate_ssm(audio_data, hop_length_factor, n_chroma, bins_per_octave):
    sr, audio = audio_data
    # Ensure audio is mono for processing
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    audio = ensure_length(audio, 2048)  # Ensure audio length for n_fft=2048
    
    # Adjust hop_length based on the factor provided by the user
    hop_length = int(0.0695 * sr * hop_length_factor)
    
    X = compute_hpcp(audio, sr, hop_length, n_chroma, bins_per_octave)
    S = compute_ssm(X)
    return S

# Define sliders for the parameters
hop_length_slider = gr.Slider(minimum=0.5, maximum=10.0, step=0.1, value=1.0, label="Hop Length Factor")
n_chroma_slider = gr.Slider(minimum=12, maximum=100, step=1, value=36, label="Number of Chroma Bins")
bins_per_octave_slider = gr.Slider(minimum=36, maximum=500, step=3, value=108, label="Bins Per Octave")

iface = gr.Interface(
    fn=generate_ssm,
    inputs=[
        gr.Audio(type="numpy", label="Upload Audio"),
        hop_length_slider,
        n_chroma_slider,
        bins_per_octave_slider
    ],
    outputs=gr.Image(type="numpy"),
    title="Music Structure Analysis",
    description="Upload an audio file to generate its segment similarity matrix (SSM). Adjust the parameters to change the resolution of the SSM."
)

iface.launch(server_name="0.0.0.0")
