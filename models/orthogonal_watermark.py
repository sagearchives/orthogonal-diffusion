"""
Orthogonal Diffuse: Core Watermarking Implementation
Real implementation without fabricated results
"""
import numpy as np
from scipy.linalg import svd
from typing import Tuple, Optional, Dict
import hashlib


class OrthogonalWatermark:
    """
    Core watermarking class implementing orthogonal projection
    """
    
    def __init__(
        self,
        w_radius: int = 10,
        num_scales: int = 3,
        svd_rank: int = 50,
        image_size: int = 256
    ):
        self.w_radius = w_radius
        self.num_scales = num_scales
        self.svd_rank = svd_rank
        self.image_size = image_size
        
    def generate_ring_pattern(
        self,
        radius: float,
        width: float,
        key: int,
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """Generate single ring pattern in frequency domain"""
        h, w = shape
        # Create frequency grid
        y = np.fft.fftfreq(h)[:, None]
        x = np.fft.fftfreq(w)[None, :]
        freq_magnitude = np.sqrt(x**2 + y**2)
        
        # Create ring mask
        mask = ((freq_magnitude >= radius - width) & 
                (freq_magnitude <= radius + width)).astype(float)
        
        # Generate random phase based on key
        np.random.seed(key)
        phase = np.random.uniform(0, 2*np.pi, shape)
        
        # Create pattern
        pattern = mask * np.exp(1j * phase)
        return pattern
    
    def generate_watermark(
        self,
        key: int,
        shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Generate multi-scale ring watermark
        
        Args:
            key: Watermark key (integer seed)
            shape: (C, H, W) image shape
            
        Returns:
            Watermark pattern in spatial domain
        """
        c, h, w = shape
        watermark = np.zeros((c, h, w))
        
        for s in range(self.num_scales):
            radius = (2 ** s) * self.w_radius / max(h, w)
            width = radius * 0.1
            scale_key = key + s * 1000
            
            for ch in range(c):
                ch_key = scale_key + ch
                ring = self.generate_ring_pattern(radius, width, ch_key, (h, w))
                
                # Inverse FFT to spatial domain
                spatial = np.fft.ifft2(ring).real
                watermark[ch] += spatial * (1.0 / self.num_scales)
        
        # Normalize
        watermark = watermark / (np.abs(watermark).max() + 1e-8)
        return watermark
    
    def orthogonal_project(
        self,
        watermark: np.ndarray,
        subspace_basis: np.ndarray
    ) -> np.ndarray:
        """
        Project watermark onto orthogonal complement of subspace
        
        Args:
            watermark: Watermark pattern [C, H, W]
            subspace_basis: Subspace basis vectors [d, k]
            
        Returns:
            Orthogonally projected watermark
        """
        # Flatten watermark
        w_flat = watermark.flatten()
        
        # Project onto subspace
        projection = subspace_basis @ (subspace_basis.T @ w_flat)
        
        # Orthogonal complement
        w_orth = w_flat - projection
        
        # Reshape back
        return w_orth.reshape(watermark.shape)
    
    def detect_watermark(
        self,
        image: np.ndarray,
        key: int
    ) -> Tuple[float, Dict]:
        """
        Detect watermark in image using correlation
        
        Args:
            image: Image to check [C, H, W]
            key: Watermark key to test
            
        Returns:
            correlation_score: Detection score
            details: Dictionary with per-scale scores
        """
        c, h, w = image.shape
        
        # Generate reference watermark
        reference = self.generate_watermark(key, (c, h, w))
        
        # Compute correlation in frequency domain
        image_fft = np.fft.fft2(image, axes=(1, 2))
        ref_fft = np.fft.fft2(reference, axes=(1, 2))
        
        scores = []
        for s in range(self.num_scales):
            radius = (2 ** s) * self.w_radius / max(h, w)
            width = radius * 0.1
            
            # Create ring mask
            y = np.fft.fftfreq(h)[:, None]
            x = np.fft.fftfreq(w)[None, :]
            freq_mag = np.sqrt(x**2 + y**2)
            mask = ((freq_mag >= radius - width) & 
                   (freq_mag <= radius + width))
            
            # Correlation in this band
            corr = np.abs(image_fft * np.conj(ref_fft) * mask).sum()
            norm = (np.abs(image_fft * mask).sum() * 
                   np.abs(ref_fft * mask).sum()) ** 0.5
            
            score = corr / (norm + 1e-8)
            scores.append(float(score))
        
        avg_score = float(np.mean(scores))
        
        return avg_score, {
            'scale_scores': scores,
            'average': avg_score
        }


class ScoreFunctionAnalyzer:
    """Analyzes score function geometry"""
    
    def __init__(self, svd_rank: int = 50):
        self.svd_rank = svd_rank
    
    def estimate_subspace(
        self,
        perturbations: np.ndarray,
        responses: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate score function subspace from perturbation responses
        
        Args:
            perturbations: Perturbation samples [N, d]
            responses: Score function responses [N, d]
            
        Returns:
            U_k: Top-k singular vectors [d, k]
            singular_values: All singular values
        """
        # Center responses
        mean_response = responses.mean(axis=0)
        centered = responses - mean_response
        
        # SVD
        U, S, Vt = svd(centered.T, full_matrices=False)
        
        # Extract top-k
        U_k = U[:, :self.svd_rank]
        
        return U_k, S
    
    def compute_spectral_gap(self, singular_values: np.ndarray) -> float:
        """Compute spectral gap between signal and noise subspace"""
        if len(singular_values) <= self.svd_rank:
            return 0.0
        return float(singular_values[self.svd_rank - 1] / 
                    (singular_values[self.svd_rank] + 1e-8))
