"""
Experiment Runner: Real Watermarking Experiments
Uses models from models/ and data from data/
"""
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))

from orthogonal_watermark import OrthogonalWatermark, ScoreFunctionAnalyzer


class WatermarkExperiment:
    """Runs comprehensive watermarking experiments"""
    
    def __init__(self, output_dir: str = 'experiments/results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.watermarker = OrthogonalWatermark(
            w_radius=10,
            num_scales=3,
            svd_rank=50,
            image_size=256
        )
        
    def load_synthetic_images(self, data_dir: str, num_images: int = 100) -> np.ndarray:
        """Load synthetic images from data directory"""
        data_path = Path(data_dir)
        
        # Load metadata
        metadata_file = data_path / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load images
        images = []
        for img_info in metadata['images'][:num_images]:
            img_path = data_path / img_info['filename']
            img = np.load(img_path)
            images.append(img)
        
        return np.array(images)
    
    def apply_attack(
        self,
        image: np.ndarray,
        attack_type: str,
        **params
    ) -> np.ndarray:
        """Apply various attacks to watermarked images"""
        if attack_type == 'jpeg_compression':
            quality = params.get('quality', 75)
            # Simulate JPEG by adding quantization noise
            attacked = image + np.random.normal(0, (100-quality)/1000, image.shape)
            return np.clip(attacked, 0, 1)
        
        elif attack_type == 'gaussian_blur':
            sigma = params.get('sigma', 1.0)
            # Apply Gaussian blur in frequency domain
            c, h, w = image.shape
            result = np.zeros_like(image)
            for ch in range(c):
                freq = np.fft.fft2(image[ch])
                y = np.fft.fftfreq(h)[:, None]
                x = np.fft.fftfreq(w)[None, :]
                freq_mag = np.sqrt(x**2 + y**2)
                gaussian_filter = np.exp(-freq_mag**2 / (2 * sigma**2))
                filtered = freq * gaussian_filter
                result[ch] = np.fft.ifft2(filtered).real
            return result
        
        elif attack_type == 'crop':
            crop_ratio = params.get('ratio', 0.8)
            c, h, w = image.shape
            new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            cropped = image[:, start_h:start_h+new_h, start_w:start_w+new_w]
            # Simple resize back
            from scipy.ndimage import zoom
            zoom_factors = [1, h/cropped.shape[1], w/cropped.shape[2]]
            return zoom(cropped, zoom_factors, order=1)
        
        elif attack_type == 'noise':
            noise_level = params.get('level', 0.02)
            noise = np.random.normal(0, noise_level, image.shape)
            return np.clip(image + noise, 0, 1)
        
        else:
            return image
    
    def run_robustness_experiment(
        self,
        watermark_key: int = 42
    ) -> dict:
        """Run robustness experiments with real attacks"""
        print(f"Loading synthetic images...")
        images = self.load_synthetic_images('data/synthetic/watermarked_test', 100)
        
        # Embed watermarks
        print("Embedding watermarks...")
        watermarked = []
        for img in images:
            wm = self.watermarker.generate_watermark(watermark_key, img.shape)
            embedded = img + wm * 0.05
            watermarked.append(np.clip(embedded, 0, 1))
        
        watermarked = np.array(watermarked)
        
        # Test attacks
        attacks = {
            'jpeg_q75': {'type': 'jpeg_compression', 'quality': 75},
            'jpeg_q50': {'type': 'jpeg_compression', 'quality': 50},
            'blur_1.0': {'type': 'gaussian_blur', 'sigma': 1.0},
            'blur_1.5': {'type': 'gaussian_blur', 'sigma': 1.5},
            'crop_80': {'type': 'crop', 'ratio': 0.8},
            'crop_70': {'type': 'crop', 'ratio': 0.7},
            'noise_0.01': {'type': 'noise', 'level': 0.01},
            'noise_0.02': {'type': 'noise', 'level': 0.02}
        }
        
        results = {}
        
        for attack_name, attack_params in attacks.items():
            print(f"Testing attack: {attack_name}...")
            
            detection_scores = []
            for wm_img in watermarked:
                # Apply attack
                attacked = self.apply_attack(
                    wm_img,
                    attack_params['type'],
                    **{k: v for k, v in attack_params.items() if k != 'type'}
                )
                
                # Detect watermark
                score, details = self.watermarker.detect_watermark(
                    attacked,
                    watermark_key
                )
                detection_scores.append(score)
            
            # Compute statistics
            scores_array = np.array(detection_scores)
            results[attack_name] = {
                'mean_score': float(scores_array.mean()),
                'std_score': float(scores_array.std()),
                'min_score': float(scores_array.min()),
                'max_score': float(scores_array.max()),
                'detection_rate': float((scores_array > 0.5).mean())
            }
        
        # Test on non-watermarked images (false positives)
        print("Testing false positive rate...")
        non_watermarked = self.load_synthetic_images('data/synthetic/non_watermarked_test', 100)
        
        fp_scores = []
        for img in non_watermarked:
            score, _ = self.watermarker.detect_watermark(img, watermark_key)
            fp_scores.append(score)
        
        results['false_positive_analysis'] = {
            'mean_score': float(np.mean(fp_scores)),
            'std_score': float(np.std(fp_scores)),
            'false_positive_rate_0.5': float((np.array(fp_scores) > 0.5).mean())
        }
        
        return results
    
    def run_quality_analysis(
        self,
        watermark_key: int = 42
    ) -> dict:
        """Analyze image quality metrics"""
        print(f"Running quality analysis...")
        images = self.load_synthetic_images('data/synthetic/watermarked_test', 100)
        
        psnr_values = []
        mse_values = []
        
        for img in images:
            wm = self.watermarker.generate_watermark(watermark_key, img.shape)
            watermarked = img + wm * 0.05
            watermarked = np.clip(watermarked, 0, 1)
            
            # Compute MSE
            mse = np.mean((img - watermarked) ** 2)
            mse_values.append(mse)
            
            # Compute PSNR
            if mse > 0:
                psnr = 10 * np.log10(1.0 / mse)
            else:
                psnr = 100
            psnr_values.append(psnr)
        
        return {
            'psnr_db': {
                'mean': float(np.mean(psnr_values)),
                'std': float(np.std(psnr_values)),
                'min': float(np.min(psnr_values)),
                'max': float(np.max(psnr_values))
            },
            'mse': {
                'mean': float(np.mean(mse_values)),
                'std': float(np.std(mse_values))
            },
            'num_images': len(images)
        }
    
    def save_results(self, results: dict, filename: str):
        """Save results to JSON file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


def main():
    """Run all experiments"""
    experiment = WatermarkExperiment()
    
    # Run robustness experiments
    print("\n" + "="*60)
    print("Running Robustness Experiments")
    print("="*60)
    robustness_results = experiment.run_robustness_experiment()
    experiment.save_results(robustness_results, 'robustness_results.json')
    
    # Run quality analysis
    print("\n" + "="*60)
    print("Running Quality Analysis")
    print("="*60)
    quality_results = experiment.run_quality_analysis()
    experiment.save_results(quality_results, 'quality_results.json')
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    main()
