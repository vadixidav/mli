use image::ImageResult;
use mli::{Forward, Graph, Train};
use mli_conv::Conv2;
use mli_ndarray::Map2Static;
use mli_relu::Softplus;
use mli_sigmoid::Logistic;
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
    let down = Conv2(down_filter).forward(&image).1;
    let right = Conv2(right_filter).forward(&image).1;
    (down.map(|f| f.powi(2)) + right.map(|f| f.powi(2))).map(|f| f.sqrt())
}

fn open_image(path: impl AsRef<Path>) -> ImageResult<Array2<f32>> {
    let image = open_gray_image(path)?;
    Ok(image.map(|&n| f32::from(n) / 255.0))
}

fn save_image(path: impl AsRef<Path>, image: &Array2<f32>) -> ImageResult<()> {
    let image = image.map(|&n| num::clamp(n * 255.0, 0.0, 255.0) as u8);
    save_gray_image(path, image.view())
}

fn main() -> ImageResult<()> {
    let opt = Opt::from_args();
    let mut prng = Pcg64::from_seed([0; 32]);
    let image = open_image(opt.file)?;
    let sobel_image = sobel(&image);
    let conv_layers = 4;
    let filter_radius = 1usize;
    let filter_len = (filter_radius * 2 + 1).pow(2);
    let padding = (conv_layers * filter_radius - 1) as i32;
    #[allow(clippy::deref_addrof)]
    let expected = sobel_image.slice(s![padding..-padding, padding..-padding]);
    save_image(opt.output_dir.join("actual.png"), &expected.to_owned())?;
    let mut random_filter = |mean: f32, variance: f32| {
        // Xavier initialize by changing the variance to be 1/N where N is the number of neurons.
        Array::from_iter(
            Normal::new(mean / filter_len as f32, variance / filter_len as f32)
                .unwrap()
                .sample_iter(&mut prng)
                .take(filter_len),
        )
        .into_shape((filter_radius * 2 + 1, filter_radius * 2 + 1))
        .unwrap()
    };
    let mut train_filter = Conv2(random_filter(0.0, 3.0))
        .chain(Map2Static(Softplus))
        .chain(Conv2(random_filter(0.0, 3.0)))
        .chain(Map2Static(Logistic))
        .chain(Conv2(random_filter(0.0, 3.0)))
        .chain(Map2Static(Logistic))
        .chain(Conv2(random_filter(15.0, 15.0)));
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
