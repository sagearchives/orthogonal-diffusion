"""
Data Generation for Watermarking Experiments
Generates procedural test images with reproducible seeds
"""
import numpy as np
import os
from pathlib import Path
import json


class ImageGenerator:
    """Generates test images using frequency-domain synthesis"""
    
    def __init__(self, image_size: int = 256, num_channels: int = 3):
        self.image_size = image_size
        self.num_channels = num_channels
    
    def generate_image(self, seed: int) -> np.ndarray:
        """
        Generate single synthetic image
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Image array of shape [C, H, W] in range [0, 1]
        """
        np.random.seed(seed)
        img = np.zeros((self.num_channels, self.image_size, self.image_size))
        
        for c in range(self.num_channels):
            # Generate frequency components
            freq = np.fft.fft2(np.random.randn(self.image_size, self.image_size))
            
            # Apply low-pass filter for natural appearance
            y = np.fft.fftfreq(self.image_size)[:, None]
            x = np.fft.fftfreq(self.image_size)[None, :]
            freq_mag = np.sqrt(x**2 + y**2)
            filter_mask = np.exp(-freq_mag * 10)
            
            # Filter and convert to spatial
            filtered = freq * filter_mask
            spatial = np.fft.ifft2(filtered).real
            
            # Normalize to [0, 1]
            spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)
            img[c] = spatial
        
        return img
    
    def generate_dataset(
        self,
        num_images: int,
        start_seed: int,
        output_dir: str
    ) -> dict:
        """
        Generate dataset of synthetic images
        
        Args:
            num_images: Number of images to generate
            start_seed: Starting seed value
            output_dir: Directory to save images
            
        Returns:
            metadata: Dictionary with generation info
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'num_images': num_images,
            'image_size': self.image_size,
            'num_channels': self.num_channels,
            'start_seed': start_seed,
            'images': []
        }
        
        for i in range(num_images):
            seed = start_seed + i
            image = self.generate_image(seed)
            
            # Save image
            filename = f'synthetic_{seed:05d}.npy'
            filepath = output_path / filename
            np.save(filepath, image)
            
            metadata['images'].append({
                'index': i,
                'seed': seed,
                'filename': filename,
                'shape': list(image.shape)
            })
        
        # Save metadata
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Generated {num_images} synthetic images in {output_dir}")
        print(f"Metadata saved to {metadata_path}")
        
        return metadata


def main():
    """Generate datasets for experiments"""
    generator = ImageGenerator(image_size=256, num_channels=3)
    
    # Generate watermarked test set
    print("Generating watermarked test set...")
    generator.generate_dataset(
        num_images=100,
        start_seed=42,
        output_dir='data/synthetic/watermarked_test'
    )
    
    # Generate non-watermarked test set  
    print("\nGenerating non-watermarked test set...")
    generator.generate_dataset(
        num_images=100,
        start_seed=999,
        output_dir='data/synthetic/non_watermarked_test'
    )
    
    print("\n✓ Dataset generation complete!")


if __name__ == '__main__':
    main()
