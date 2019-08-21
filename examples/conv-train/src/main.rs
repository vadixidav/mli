use image::ImageResult;
use mli::{Forward, Train};
use mli_conv::Conv2;
use ndarray::{array, Array2};
use ndarray_image::{open_gray_image, save_gray_image};
use std::path::{Path, PathBuf};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "image", about = "Loads and convolves an image as a test")]
struct Opt {
    /// Number of training runs
    #[structopt(short = "t", default_value = "50")]
    train_runs: usize,
    /// Learning rate
    #[structopt(short = "l", default_value = "0.001")]
    learning_rate: f32,
    /// File to load
    #[structopt(parse(from_os_str))]
    file: PathBuf,
    /// Output directory
    #[structopt(parse(from_os_str))]
    output_dir: PathBuf,
}

fn sobel(image: &Array2<f32>) -> Array2<f32> {
    let down_filter = array![[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0],];
    let right_filter = down_filter.t().to_owned();
    fn square(f: &f32) -> f32 {
        f.powi(2)
    }
    fn sqrt(f: &f32) -> f32 {
        f.sqrt()
    }
    let down = Conv2(down_filter).forward(&image).1;
    let right = Conv2(right_filter).forward(&image).1;
    (down.map(square) + right.map(square)).map(sqrt)
}

fn open_image(path: impl AsRef<Path>) -> ImageResult<Array2<f32>> {
    let image = open_gray_image(path)?;
    Ok(image.map(|&n| n as f32))
}

fn save_image(path: impl AsRef<Path>, image: &Array2<f32>) -> ImageResult<()> {
    let image = image.map(|&n| num::clamp(n, 0.0, 255.0) as u8);
    save_gray_image(path, image.view())
}

fn main() -> ImageResult<()> {
    let opt = Opt::from_args();
    let image = open_image(opt.file)?;
    let sobel_image = sobel(&image);
    let mut train_filter = Conv2(array![[1.5, -2.0, 0.1], [1.0, 0.5, 0.4], [1.4, 1.1, -1.2]]);
    for i in 0..opt.train_runs {
        let (internal, output) = train_filter.forward(&image);
        save_image(opt.output_dir.join(format!("iteration{}.png", i)), &output)?;
        // The loss function is (n - t)^2, so 2*(n - t) is dE/df where
        // `E` is loss and `f` is output.
        let output_delta = -opt.learning_rate * (output - sobel_image.view()).map(|n| 2.0 * n);
        train_filter.propogate(&image, &internal, &output_delta);
    }
    Ok(())
}
