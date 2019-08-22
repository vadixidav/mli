use image::ImageResult;
use mli::{Forward, Graph, Train};
use mli_conv::Conv2;
use mli_ndarray::Map2Many;
use mli_relu::ReluSoftplus;
use ndarray::{array, s, Array, Array2};
use ndarray_image::{open_gray_image, save_gray_image};
use std::path::{Path, PathBuf};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "image", about = "Loads and convolves an image as a test")]
struct Opt {
    /// Number of epochs
    #[structopt(short = "t", default_value = "50")]
    epochs: usize,
    /// Learning rate
    #[structopt(short = "i", default_value = "0.000000000001")]
    initial_learning_rate: f32,
    /// Learning rate after epoch
    #[structopt(short = "l", default_value = "0.000000001")]
    learning_rate_after_10: f32,
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
    save_image(opt.output_dir.join("actual.png"), &sobel_image)?;
    let shapes = [
        (image.shape()[0] - 2, image.shape()[1] - 2),
        (image.shape()[0] - 4, image.shape()[1] - 4),
    ];
    let mut train_filter = Conv2(array![[-1.0, -1.0, -1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        .chain(Map2Many(Array::from_elem(shapes[0], ReluSoftplus)))
        .chain(Conv2(array![
            [-1.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0]
        ]))
        .chain(Map2Many(Array::from_elem(shapes[1], ReluSoftplus)));
    let expected = sobel_image.slice(s![1..-1, 1..-1]);
    for i in 0..opt.epochs {
        let (internal, output) = train_filter.forward(&image);
        save_image(opt.output_dir.join(format!("iteration{}.png", i)), &output)?;
        eprintln!("Filter 1:\n{:?}", ((train_filter.0).0).0);
        eprintln!("Filter 2:\n{:?}", (train_filter.0).1);
        // The loss function is (n - t)^2, so 2*(n - t) is dE/df where
        // `E` is loss and `f` is output.
        let output_delta = -if i < 10 {
            opt.initial_learning_rate
        } else {
            opt.learning_rate_after_10
        } * (output - expected).map(|n| 2.0 * n);
        train_filter.propogate(&image, &internal, &output_delta);
    }
    Ok(())
}
