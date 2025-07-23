"""Test script to validate JAX-accelerated normalization statistics computation.

This script compares the output of the JAX-accelerated version with the original
implementation to ensure correctness and compatibility.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

import openpi.shared.normalize as normalize
from scripts.compute_norm_stats_jax import JaxRunningStats, FastRunningStats


def test_jax_running_stats_initialization():
    """Test that JaxRunningStats initializes correctly."""
    stats = JaxRunningStats()
    assert stats._count == 0
    assert stats._mean is None
    assert stats._initialized is False


def test_fast_running_stats_initialization():
    """Test that FastRunningStats initializes correctly."""
    stats = FastRunningStats()
    assert stats.count == 0
    assert stats.sum is None
    assert stats.min_vals is None


def test_jax_running_stats_single_batch():
    """Test JaxRunningStats with a single batch of data."""
    stats = JaxRunningStats()
    
    # Create test data
    batch = np.random.randn(100, 5)  # 100 samples, 5 features
    
    # Update stats
    stats.update(batch)
    
    # Check that stats were computed correctly
    assert stats._count == 100
    assert stats._mean.shape == (5,)
    assert stats._mean_of_squares.shape == (5,)
    assert stats._min.shape == (5,)
    assert stats._max.shape == (5,)
    
    # Verify mean computation
    expected_mean = np.mean(batch, axis=0)
    np.testing.assert_allclose(stats._mean, expected_mean, rtol=1e-6)
    
    # Verify min/max computation
    expected_min = np.min(batch, axis=0)
    expected_max = np.max(batch, axis=0)
    np.testing.assert_allclose(stats._min, expected_min, rtol=1e-6)
    np.testing.assert_allclose(stats._max, expected_max, rtol=1e-6)


def test_fast_running_stats_single_batch():
    """Test FastRunningStats with a single batch of data."""
    stats = FastRunningStats()
    
    # Create test data
    batch = np.random.randn(100, 5)  # 100 samples, 5 features
    
    # Update stats
    stats.update(batch)
    
    # Check that stats were computed correctly
    assert stats.count == 100
    assert stats.sum.shape == (5,)
    assert stats.sum_sq.shape == (5,)
    assert stats.min_vals.shape == (5,)
    assert stats.max_vals.shape == (5,)
    
    # Verify sum computation
    expected_sum = np.sum(batch, axis=0)
    np.testing.assert_allclose(stats.sum, expected_sum, rtol=1e-6)
    
    # Verify min/max computation
    expected_min = np.min(batch, axis=0)
    expected_max = np.max(batch, axis=0)
    np.testing.assert_allclose(stats.min_vals, expected_min, rtol=1e-6)
    np.testing.assert_allclose(stats.max_vals, expected_max, rtol=1e-6)


def test_jax_running_stats_multiple_batches():
    """Test JaxRunningStats with multiple batches of data."""
    stats = JaxRunningStats()
    
    # Create multiple batches
    batches = [
        np.random.randn(50, 3),
        np.random.randn(75, 3),
        np.random.randn(25, 3),
    ]
    
    # Update stats with each batch
    for batch in batches:
        stats.update(batch)
    
    # Combine all batches for ground truth
    all_data = np.vstack(batches)
    
    # Get final statistics
    final_stats = stats.get_statistics()
    
    # Verify mean computation
    expected_mean = np.mean(all_data, axis=0)
    np.testing.assert_allclose(final_stats.mean, expected_mean, rtol=1e-6)
    
    # Verify standard deviation computation
    expected_std = np.std(all_data, axis=0)
    np.testing.assert_allclose(final_stats.std, expected_std, rtol=1e-6)


def test_fast_running_stats_multiple_batches():
    """Test FastRunningStats with multiple batches of data."""
    stats = FastRunningStats()
    
    # Create multiple batches
    batches = [
        np.random.randn(50, 3),
        np.random.randn(75, 3),
        np.random.randn(25, 3),
    ]
    
    # Update stats with each batch
    for batch in batches:
        stats.update(batch)
    
    # Combine all batches for ground truth
    all_data = np.vstack(batches)
    
    # Get final statistics
    final_stats = stats.get_statistics()
    
    # Verify mean computation
    expected_mean = np.mean(all_data, axis=0)
    np.testing.assert_allclose(final_stats.mean, expected_mean, rtol=1e-6)
    
    # Verify standard deviation computation
    expected_std = np.std(all_data, axis=0)
    np.testing.assert_allclose(final_stats.std, expected_std, rtol=1e-6)


def test_jax_running_stats_quantiles():
    """Test that JaxRunningStats computes quantiles correctly."""
    stats = JaxRunningStats()
    
    # Create test data with known distribution
    np.random.seed(42)
    batch = np.random.randn(1000, 2)
    
    # Update stats
    stats.update(batch)
    
    # Get statistics
    final_stats = stats.get_statistics()
    
    # Verify quantiles are within reasonable bounds
    assert np.all(final_stats.q01 < final_stats.mean)
    assert np.all(final_stats.q99 > final_stats.mean)
    assert np.all(final_stats.q01 < final_stats.q99)


def test_fast_running_stats_quantiles():
    """Test that FastRunningStats computes approximate quantiles correctly."""
    stats = FastRunningStats()
    
    # Create test data with known distribution
    np.random.seed(42)
    batch = np.random.randn(1000, 2)
    
    # Update stats
    stats.update(batch)
    
    # Get statistics
    final_stats = stats.get_statistics()
    
    # Verify quantiles are within reasonable bounds
    assert np.all(final_stats.q01 < final_stats.mean)
    assert np.all(final_stats.q99 > final_stats.mean)
    assert np.all(final_stats.q01 < final_stats.q99)


def test_error_handling():
    """Test error handling for edge cases."""
    # Test with insufficient data
    stats = JaxRunningStats()
    batch = np.random.randn(1, 3)  # Only one sample
    stats.update(batch)
    
    # Should raise error when trying to get statistics
    with pytest.raises(ValueError, match="Cannot compute statistics for less than 2 vectors"):
        stats.get_statistics()
    
    # Test with inconsistent dimensions
    stats = JaxRunningStats()
    stats.update(np.random.randn(10, 3))
    
    with pytest.raises(ValueError, match="Expected 3 features, but got 2"):
        stats.update(np.random.randn(10, 2))


def test_1d_input_handling():
    """Test that 1D inputs are handled correctly."""
    stats = JaxRunningStats()
    
    # 1D input should be reshaped to 2D
    batch_1d = np.random.randn(100)
    stats.update(batch_1d)
    
    # Check that it was reshaped correctly
    assert stats._mean.shape == (1,)
    assert stats._count == 100
    
    # Test FastRunningStats as well
    fast_stats = FastRunningStats()
    fast_stats.update(batch_1d)
    assert fast_stats.sum.shape == (1,)
    assert fast_stats.count == 100


def test_jax_array_input():
    """Test that JAX arrays are handled correctly."""
    try:
        import jax.numpy as jnp
        
        stats = JaxRunningStats()
        batch_jax = jnp.array(np.random.randn(50, 4))
        
        # Should work without error
        stats.update(batch_jax)
        
        # Verify results
        assert stats._count == 50
        assert stats._mean.shape == (4,)
        
    except ImportError:
        pytest.skip("JAX not available")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 