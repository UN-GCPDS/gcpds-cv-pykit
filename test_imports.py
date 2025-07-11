#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly.
"""

def test_imports():
    """Test that all main components can be imported."""
    try:
        # Test main package imports
        from gcpds_cv_pykit import (
            PerformanceModels, 
            SegmentationModel_Trainer,
            DICELoss, 
            CrossEntropyLoss, 
            FocalLoss, 
            TverskyLoss,
            UNet
        )
        print("✓ Main package imports successful")
        
        # Test direct imports from baseline.losses
        from gcpds_cv_pykit.baseline.losses import DICELoss, FocalLoss
        print("✓ Direct baseline.losses imports successful")
        
        # Test loss instantiation
        dice_loss = DICELoss(smooth=1.0, reduction='mean')
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        print("✓ Loss function instantiation successful")
        
        print("\n🎉 All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()