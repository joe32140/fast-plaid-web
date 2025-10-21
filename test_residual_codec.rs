use candle_core::{Device, DType, Tensor};

mod rust {
    pub mod utils {
        pub mod residual_codec;
    }
}

use rust::utils::residual_codec::ResidualCodec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    // Create test tensors
    let centroids = Tensor::randn(0f32, 1.0, (10, 128), &device)?;
    let avg_residual = Tensor::randn(0f32, 1.0, (128,), &device)?;
    let bucket_cutoffs = Tensor::randn(0f32, 1.0, (16,), &device)?;
    let bucket_weights = Tensor::randn(0f32, 1.0, (16, 128), &device)?;
    
    // Test ResidualCodec creation
    let codec = ResidualCodec::load(
        4, // nbits
        centroids,
        avg_residual,
        Some(bucket_cutoffs),
        Some(bucket_weights),
        device,
    )?;
    
    println!("✅ ResidualCodec created successfully!");
    println!("   nbits: {}", codec.nbits);
    println!("   centroids shape: {:?}", codec.centroids.dims());
    println!("   avg_residual shape: {:?}", codec.avg_residual.dims());
    
    Ok(())
}