use image::ImageResult;
use mli::{Forward, Graph, Train};
use mli_conv::Conv2;
use mli_ndarray::Map2Many;
use mli_relu::Relu;
use ndarray::{array, s, Array, Array2};
use ndarray_image::{open_gray_image, save_gray_image};
use rand_core::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64;
use std::path::{Path, PathBuf};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "image", about = "Loads and convolves an image as a test")]
struct Opt {
    /// Number of epochs
    #[structopt(short = "t", default_value = "2000")]
    epochs: usize,
    /// Number of epochs per output image
    #[structopt(short = "s", default_value = "10")]
    show_every: usize,
    /// Initial learning rate
    #[structopt(short = "i", default_value = "0.000000000001")]
    initial_learning_rate: f32,
    /// Learning rate multiplier per epoch
    #[structopt(short = "m", default_value = "1.003")]
    learning_rate_multiplier: f32,
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
    let mut prng = Pcg64::from_seed([0; 32]);
    let image = open_image(opt.file)?;
    let sobel_image = sobel(&image);
    let expected = sobel_image.slice(s![3..-3, 3..-3]);
    save_image(opt.output_dir.join("actual.png"), &expected.to_owned())?;
    let shapes = [
        (image.shape()[0] - 2, image.shape()[1] - 2),
        (image.shape()[0] - 4, image.shape()[1] - 4),
        (image.shape()[0] - 6, image.shape()[1] - 6),
        (image.shape()[0] - 8, image.shape()[1] - 8),
    ];
    let mut random_filter = || {
        // Xavier initialize by changing the variance to be 1/N where N is the number of neurons.
        Array::from_iter(
            Normal::new(1.0 / 9.0, 1.0 / 9.0)
                .unwrap()
                .sample_iter(&mut prng)
                .take(9),
        )
        .into_shape((3, 3))
        .unwrap()
    };
    let mut train_filter = Conv2(random_filter())
        .chain(Map2Many(Array::from_elem(shapes[0], Relu)))
        .chain(Conv2(random_filter()))
        .chain(Map2Many(Array::from_elem(shapes[1], Relu)))
        .chain(Conv2(random_filter()))
        .chain(Map2Many(Array::from_elem(shapes[2], Relu)))
        .chain(Conv2(random_filter()))
        .chain(Map2Many(Array::from_elem(shapes[3], Relu)));
    let mut learn_rate = opt.initial_learning_rate;
    for i in 0..opt.epochs {
        let (internal, output) = train_filter.forward(&image);
        if i % opt.show_every == 0 {
            save_image(opt.output_dir.join(format!("epoch{:04}.png", i)), &output)?;
        }
        let loss = (output.clone() - expected.view()).map(|n| n.powi(2)).sum();
        eprintln!("epoch {:04} loss: {}, learn_rate: {}", i, loss, learn_rate);
        // The loss function is (n - t)^2, so 2*(n - t) is dE/df where
        // `E` is loss and `f` is output.
        let delta_loss = (output - expected).map(|n| 2.0 * n);
        let output_delta = -learn_rate * delta_loss;
        train_filter.propogate(&image, &internal, &output_delta);
        learn_rate *= opt.learning_rate_multiplier;
    }
    Ok(())
}
