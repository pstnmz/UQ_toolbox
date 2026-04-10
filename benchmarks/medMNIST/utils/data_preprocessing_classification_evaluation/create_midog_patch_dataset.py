"""
Extract 224x224 patches from MIDOG++ dataset images.
Creates non-overlapping patches from each image in the dataset.

Resolution matching for cross-dataset comparison
---------------------------------------------------------
PathMNIST patches: 224×224 at 0.5 µm/px → 112 µm × 112 µm physical area
MIDOG++ images: typically at 0.23-0.25 µm/px → need resampling to 0.5 µm/px

This script resamples MIDOG++ images to match PathMNIST's physical resolution
BEFORE extracting patches to avoid magnification confounds in OOD detection.
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, use a simple replacement
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable


# Target resolution for PathMNIST compatibility
TARGET_RESOLUTION = 0.5  # µm/pixel (PathMNIST standard)

# Default MIDOG++ resolutions (if not in metadata)
# Based on MIDOG++ paper: mostly 0.23-0.25 µm/px
DEFAULT_MIDOG_RESOLUTION = 0.25  # µm/pixel


def get_image_resolution(img_meta, default=DEFAULT_MIDOG_RESOLUTION):
    """
    Extract physical resolution (µm/pixel) from image metadata.
    
    Args:
        img_meta: Image metadata dict from JSON
        default: Default resolution if not found in metadata
    
    Returns:
        float: Resolution in µm/pixel
    """
    # Check common metadata fields for resolution info
    # MIDOG++ may store this as 'mpp' (microns per pixel) or 'resolution'
    if 'mpp' in img_meta:
        return float(img_meta['mpp'])
    elif 'resolution' in img_meta:
        return float(img_meta['resolution'])
    elif 'microns_per_pixel' in img_meta:
        return float(img_meta['microns_per_pixel'])
    else:
        # Use default and warn
        return default


def resample_to_target_resolution(image, source_resolution, target_resolution=TARGET_RESOLUTION):
    """
    Resample image to match target physical resolution.
    
    Uses high-quality LANCZOS resampling to avoid aliasing artifacts.
    
    Args:
        image: PIL Image
        source_resolution: Source resolution in µm/pixel
        target_resolution: Target resolution in µm/pixel (default: 0.5 for PathMNIST)
    
    Returns:
        PIL Image resampled to target resolution
    """
    # Calculate scaling factor
    # If source is 0.25 µm/px and target is 0.5 µm/px:
    #   scale = 0.25 / 0.5 = 0.5 (downsample to 50%)
    scale_factor = source_resolution / target_resolution
    
    if abs(scale_factor - 1.0) < 0.01:  # Already at target resolution (within 1%)
        return image
    
    # Calculate new dimensions
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Resample with LANCZOS for high-quality downsampling (anti-aliasing)
    # LANCZOS is best for downsampling as it includes anti-aliasing filter
    resampled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resampled


def extract_non_overlapping_patches(image, patch_size=224, num_patches=50):
    """
    Extract non-overlapping patches from an image using a grid approach.
    
    Args:
        image: PIL Image or numpy array
        patch_size: Size of square patches (default: 224)
        num_patches: Number of patches to extract (default: 50)
    
    Returns:
        List of numpy arrays, each of shape (patch_size, patch_size, 3)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGBA to RGB if necessary
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    height, width = image.shape[:2]
    
    # Calculate grid dimensions to get non-overlapping patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    total_possible_patches = num_patches_h * num_patches_w
    
    if total_possible_patches < num_patches:
        print(f"Warning: Image size ({height}x{width}) allows only {total_possible_patches} "
              f"non-overlapping {patch_size}x{patch_size} patches. Requested {num_patches}.")
        num_patches = total_possible_patches
    
    # Generate all possible patch positions in a grid
    all_positions = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y = i * patch_size
            x = j * patch_size
            all_positions.append((y, x))
    
    # Randomly select num_patches positions
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    selected_positions = rng.choice(len(all_positions), size=num_patches, replace=False)
    
    patches = []
    for idx in selected_positions:
        y, x = all_positions[idx]
        patch = image[y:y+patch_size, x:x+patch_size]
        patches.append(patch)
    
    return patches


def create_patch_dataset(
    images_dir,
    json_path,
    output_path,
    patch_size=224,
    num_patches=50,
    save_individual=True,
    target_resolution=TARGET_RESOLUTION,
    assume_default_resolution=False
):
    """
    Create a dataset of patches from MIDOG++ images and save as .npz file.
    
    CRITICAL: Applies resolution matching to ensure patches represent the same
    physical tissue area as PathMNIST (0.5 µm/px standard).
    
    Args:
        images_dir: Path to directory containing TIFF images
        json_path: Path to MIDOGpp.json file
        output_path: Path to save .npz file
        patch_size: Size of square patches (default: 224)
        num_patches: Number of patches per image
        save_individual: If True, also save individual PNG files
        target_resolution: Target resolution in µm/px (default: 0.5 for PathMNIST)
        assume_default_resolution: If True, use default resolution for all images
                                   If False, try to read from metadata
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create directory for individual patches if needed
    if save_individual:
        patches_dir = output_path.parent / 'patches_individual'
        patches_dir.mkdir(parents=True, exist_ok=True)
    else:
        patches_dir = None
    
    # Load JSON metadata
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter images to only include those with "canine" in tumor_type
    all_images = data['images']
    canine_images = [img for img in all_images if 'canine' in img.get('tumor_type', '').lower()]
    
    # Create mapping of image_id to filename
    image_info = {img['id']: img for img in canine_images}
    
    # Extract unique tumor types from canine images
    tumor_types = sorted(list(set([img['tumor_type'] for img in canine_images])))
    tumor_type_to_label = {tumor_type: idx for idx, tumor_type in enumerate(tumor_types)}
    
    print(f"Total images in JSON: {len(all_images)}")
    print(f"Images with 'canine' tumor type: {len(image_info)}")
    print(f"\nTumor types found:")
    for tumor_type, label in tumor_type_to_label.items():
        count = sum(1 for img in canine_images if img['tumor_type'] == tumor_type)
        print(f" {label}: {tumor_type} ({count} images)")
    print(f"\n{'='*70}")
    print(f"RESOLUTION MATCHING FOR CROSS-DATASET COMPARISON")
    print(f"{'='*70}")
    print(f"Target resolution: {target_resolution} µm/px (PathMNIST standard)")
    print(f"Default MIDOG++ resolution: {DEFAULT_MIDOG_RESOLUTION} µm/px (if not in metadata)")
    print(f"Physical area per patch: {patch_size * target_resolution:.1f} µm × {patch_size * target_resolution:.1f} µm")
    print(f"{'='*70}")
    print(f"\nExtracting {num_patches} patches of size {patch_size}x{patch_size} from each image")
    print(f"Output file: {output_path}")
    if save_individual:
        print(f"Individual patches will be saved to: {patches_dir}")
    
    # Storage for patches
    all_patches = []
    all_labels = []
    all_ids = []
    failed_images = []
    resolution_stats = {}  # Track resolution per image for reporting
    
    # Process each image
    for image_id, img_meta in tqdm(image_info.items(), desc="Processing images"):
        filename = img_meta['file_name']
        tumor_type = img_meta['tumor_type']
        label = tumor_type_to_label[tumor_type]
        image_path = images_dir / filename
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            failed_images.append(filename)
            continue
        
        try:
            # Load image
            image = Image.open(image_path)
            original_size = image.size
            
            # Get source resolution
            if assume_default_resolution:
                source_resolution = DEFAULT_MIDOG_RESOLUTION
            else:
                source_resolution = get_image_resolution(img_meta, default=DEFAULT_MIDOG_RESOLUTION)
            
            # Track resolution for reporting
            if source_resolution not in resolution_stats:
                resolution_stats[source_resolution] = 0
            resolution_stats[source_resolution] += 1
            
            # CRITICAL: Resample to target resolution BEFORE extracting patches
            # This ensures physical tissue area is consistent with PathMNIST
            image = resample_to_target_resolution(image, source_resolution, target_resolution)
            resampled_size = image.size
            
            # Extract patches from resampled image
            patches = extract_non_overlapping_patches(
                image, 
                patch_size=patch_size, 
                num_patches=num_patches
            )
            
            # Store patches with labels and IDs
            image_stem = Path(filename).stem  # e.g., "201" from "201.tiff"
            
            for patch_idx, patch in enumerate(patches):
                # Create ID: imageid_patch{nb}
                patch_id = f"{image_stem}_patch{patch_idx}"
                
                # Add to collections
                all_patches.append(patch)
                all_labels.append(label)
                all_ids.append(patch_id)
                
                # Save individual PNG if requested
                if save_individual:
                    patch_filename = f"{patch_id}.png"
                    patch_path = patches_dir / patch_filename
                    patch_img = Image.fromarray(patch.astype(np.uint8))
                    patch_img.save(patch_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_images.append(filename)
    
    # Convert to numpy arrays
    print(f"\nConverting to numpy arrays...")
    patches_array = np.array(all_patches, dtype=np.uint8)
    labels_array = np.array(all_labels, dtype=np.int32)
    ids_array = np.array(all_ids, dtype=object)
    
    print(f"Patches array shape: {patches_array.shape}")
    print(f"Labels array shape: {labels_array.shape}")
    print(f"IDs array shape: {ids_array.shape}")
    
    # Save as .npz file
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        images=patches_array,
        labels=labels_array,
        ids=ids_array,
        tumor_types=list(tumor_type_to_label.keys())
    )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"Total patches created: {len(all_patches)}")
    print(f"Successfully processed: {len(image_info) - len(failed_images)}/{len(image_info)} images")
    print(f"\nResolution matching applied:")
    print(f" Target resolution: {target_resolution} µm/px")
    print(f" Source resolutions found:")
    for res, count in sorted(resolution_stats.items()):
        scale = res / target_resolution
        print(f" {res} µm/px: {count} images (resampled by {scale:.3f}x)")
    print(f" Physical patch size: {patch_size * target_resolution:.1f} µm × {patch_size * target_resolution:.1f} µm")
    print(f"\nFile saved to: {output_path}")
    if save_individual:
        print(f"Individual patches saved to: {patches_dir}")
    
    if failed_images:
        print(f"\nFailed images ({len(failed_images)}):")
        for fname in failed_images:
            print(f" - {fname}")
    
    # Save metadata as separate JSON file
    metadata = {
        'patch_size': patch_size,
        'num_patches_per_image': num_patches,
        'total_patches': len(all_patches),
        'total_images': len(image_info) - len(failed_images),
        'filter_applied': 'canine tumor_type only',
        'canine_images_found': len(image_info),
        'tumor_types': tumor_type_to_label,
        'resolution_matching': {
            'applied': True,
            'target_resolution_um_per_px': target_resolution,
            'default_source_resolution_um_per_px': DEFAULT_MIDOG_RESOLUTION,
            'source_resolutions_found': {str(k): v for k, v in resolution_stats.items()},
            'physical_patch_size_um': patch_size * target_resolution,
            'note': 'Images resampled to match PathMNIST physical resolution before patch extraction'
        },
        'pathmnist_compatibility': {
            'resolution': f'{target_resolution} µm/px',
            'patch_size': f'{patch_size}x{patch_size} pixels',
            'physical_area': f'{patch_size * target_resolution:.1f} µm × {patch_size * target_resolution:.1f} µm',
            'note': 'Patches represent same physical tissue area as PathMNIST'
        },
        'individual_patches_saved': save_individual,
        'individual_patches_dir': str(patches_dir) if save_individual else None,
        'failed_images': failed_images,
        'source_json': str(json_path),
        'source_images_dir': str(images_dir)
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract non-overlapping patches from MIDOG++ images and save as .npz'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default='/home/lito/Documents/Codes/UQ_toolbox/benchmarks/medMNIST/Data/MIDOG++/images',
        help='Directory containing TIFF images'
    )
    parser.add_argument(
        '--json-path',
        type=str,
        default='/home/lito/Documents/Codes/UQ_toolbox/benchmarks/medMNIST/Data/MIDOG++/images/MIDOGpp.json',
        help='Path to MIDOGpp.json metadata file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/home/lito/Documents/Codes/UQ_toolbox/benchmarks/medMNIST/Data/MIDOG++/midog_canine_patches.npz',
        help='Path to save .npz file'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=224,
        help='Size of square patches (default: 224)'
    )
    parser.add_argument(
        '--num-patches',
        type=int,
        default=50,
        help='Number of patches to extract per image (default: 50)'
    )
    parser.add_argument(
        '--save-individual',
        action='store_true',
        default=True,
        help='Save individual PNG files in addition to .npz (default: True)'
    )
    parser.add_argument(
        '--no-save-individual',
        dest='save_individual',
        action='store_false',
        help='Do not save individual PNG files'
    )
    parser.add_argument(
        '--target-resolution',
        type=float,
        default=TARGET_RESOLUTION,
        help=f'Target resolution in µm/px (default: {TARGET_RESOLUTION} for PathMNIST compatibility)'
    )
    parser.add_argument(
        '--assume-default-resolution',
        action='store_true',
        default=False,
        help=f'Assume all images are at {DEFAULT_MIDOG_RESOLUTION} µm/px (ignore metadata)'
    )
    
    args = parser.parse_args()
    
    create_patch_dataset(
        images_dir=args.images_dir,
        json_path=args.json_path,
        output_path=args.output_path,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        save_individual=args.save_individual,
        target_resolution=args.target_resolution,
        assume_default_resolution=args.assume_default_resolution
    )


if __name__ == '__main__':
    main()
