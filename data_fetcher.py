# !/usr/bin/env python3
"""
ArcGIS World Imagery Satellite Image Fetcher
Downloads satellite imagery for properties using lat/long coordinates
"""

import pandas as pd
import numpy as np
import requests
import os
import time
from pathlib import Path
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArcGISImageFetcher:
    """Fetches satellite images from ArcGIS World Imagery service"""
    
    BASE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    
    def __init__(self, 
                 image_size: int = 256,
                 meters_per_pixel: float = 0.5,
                 output_dir: str = "satellite_images"):
        """
        Initialize the image fetcher
        
        Args:
            image_size: Size of output image in pixels (width and height)
            meters_per_pixel: Resolution - meters represented per pixel
            output_dir: Directory to save downloaded images
        """
        self.image_size = image_size
        self.meters_per_pixel = meters_per_pixel
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ArcGIS Image Fetcher")
        logger.info(f"Image size: {image_size}x{image_size} pixels")
        logger.info(f"Resolution: {meters_per_pixel}m per pixel")
        logger.info(f"Output directory: {output_dir}")
    
    def lat_lon_to_bbox(self, lat: float, lon: float) -> Tuple[float, float, float, float]:
        """
        Convert lat/lon to bounding box for ArcGIS API
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple of (xmin, ymin, xmax, ymax) in EPSG:4326
        """
        # Calculate the extent in degrees
        # Approximate: 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        
        extent_meters = (self.image_size / 2) * self.meters_per_pixel
        
        lat_degree_per_meter = 1.0 / 111320.0
        lon_degree_per_meter = 1.0 / (111320.0 * abs(np.cos(np.radians(lat))))
        
        delta_lat = extent_meters * lat_degree_per_meter
        delta_lon = extent_meters * lon_degree_per_meter
        
        xmin = lon - delta_lon
        ymin = lat - delta_lat
        xmax = lon + delta_lon
        ymax = lat + delta_lat
        
        return xmin, ymin, xmax, ymax
    
    def fetch_image(self, lat: float, lon: float, save_path: str) -> bool:
        """
        Fetch a single satellite image
        
        Args:
            lat: Latitude
            lon: Longitude
            save_path: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate bounding box
            xmin, ymin, xmax, ymax = self.lat_lon_to_bbox(lat, lon)
            
            # Construct API URL
            params = {
                'bbox': f"{xmin},{ymin},{xmax},{ymax}",
                'bboxSR': '4326',  # WGS84 coordinate system
                'size': f"{self.image_size},{self.image_size}",
                'format': 'png',
                'f': 'image'
            }
            
            # Make request
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Save image
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch image for lat={lat}, lon={lon}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error for lat={lat}, lon={lon}: {e}")
            return False
    
    def fetch_dataset_images(self, 
                            csv_path: str, 
                            dataset_type: str = 'train',
                            delay: float = 0.1,
                            max_images: int = None) -> pd.DataFrame:
        """
        Fetch images for entire dataset
        
        Args:
            csv_path: Path to Excel file with lat/long data
            dataset_type: 'train' or 'test'
            delay: Delay between requests in seconds (to be respectful)
            max_images: Maximum number of images to download (for testing)
            
        Returns:
            DataFrame with added 'image_path' column
        """
        logger.info(f"Loading dataset from {csv_path}")
        df = pd.read_excel(csv_path)
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Determine output directory
        output_dir = self.train_dir if dataset_type == 'train' else self.test_dir
        
        # Limit dataset if specified
        if max_images:
            df = df.head(max_images)
            logger.info(f"Limited to {max_images} images for testing")
        
        # Add image path column
        df['image_path'] = ''
        
        success_count = 0
        fail_count = 0
        
        logger.info(f"Starting to download {len(df)} images...")
        
        for idx, row in df.iterrows():
            property_id = row['id']
            lat = row['lat']
            lon = row['long']
            
            # Create filename using property ID
            filename = f"{property_id}.png"
            save_path = output_dir / filename
            
            # Skip if already exists
            if save_path.exists():
                df.at[idx, 'image_path'] = str(save_path)
                logger.info(f"[{idx+1}/{len(df)}] Skipped (already exists): {filename}")
                success_count += 1
                continue
            
            # Fetch image
            if self.fetch_image(lat, lon, str(save_path)):
                df.at[idx, 'image_path'] = str(save_path)
                success_count += 1
                logger.info(f"[{idx+1}/{len(df)}] Downloaded: {filename}")
            else:
                fail_count += 1
                logger.warning(f"[{idx+1}/{len(df)}] Failed: {filename}")
            
            # Respectful delay between requests
            time.sleep(delay)
            
            # Progress update every 100 images
            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: {idx+1}/{len(df)} | Success: {success_count} | Failed: {fail_count}")
        
        logger.info(f"Download complete!")
        logger.info(f"Total: {len(df)} | Success: {success_count} | Failed: {fail_count}")
        
        # Save updated dataframe with image paths
        output_csv = self.output_dir / f"{dataset_type}_with_images.csv"
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved dataset with image paths to: {output_csv}")
        
        return df


def main():
    """Main execution function"""
    
    logger.info("="*60)
    logger.info("ArcGIS World Imagery Satellite Image Fetcher")
    logger.info("="*60)
    
    # Initialize fetcher
    fetcher = ArcGISImageFetcher(
        image_size=256,          # 256x256 pixel images
        meters_per_pixel=0.5,    # 0.5 meters per pixel (high resolution)
        output_dir="satellite_images"
    )
    
    # Download training images
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Downloading TRAINING images")
    logger.info("="*60)
    train_df = fetcher.fetch_dataset_images(
        csv_path='datasets/train(1).xlsx',
        dataset_type='train',
        delay=0.1,  # 100ms delay between requests
        max_images=None  # Set to small number (e.g., 10) for testing
    )
    
    # Download test images
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Downloading TEST images")
    logger.info("="*60)
    test_df = fetcher.fetch_dataset_images(
        csv_path='datasets/test2.xlsx',
        dataset_type='test',
        delay=0.1,
        max_images=None  # Set to small number for testing
    )
    
    logger.info("\n" + "="*60)
    logger.info("ALL DONE!")
    logger.info("="*60)
    logger.info(f"Training images: {len(train_df)}")
    logger.info(f"Test images: {len(test_df)}")
    logger.info(f"Images saved in: {fetcher.output_dir}")


if __name__ == "__main__":
    main()

