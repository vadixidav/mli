use image::ImageResult;
use mli::{Backward, Forward, Graph, Train};
use mli_conv::Conv2;
use mli_ndarray::Map2One;
use mli_relu::Blu;
use ndarray::{array, s, Array, Array2, OwnedRepr};
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
    #[structopt(short = "s", default_value = "1")]
    show_every: usize,
    /// Initial learning rate
    #[structopt(short = "i", default_value = "0.000000001")]
    initial_learning_rate: f32,
    /// Learning rate multiplier per epoch
    #[structopt(short = "m", default_value = "1.003")]
    learning_rate_multiplier: f32,
    /// Seed
    #[structopt(short = "z", default_value = "42")]
    seed: u32,
    /// Beta value for NAG
    #[structopt(short = "b", default_value = "0.9")]
    beta: f32,
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
    let down = Conv2::new(down_filter).forward(&image).1;
    let right = Conv2::new(right_filter).forward(&image).1;
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
    let image = open_image(opt.file)?;
    let sobel_image = sobel(&image);
    let conv_layers = 5;
    let filter_radius = 1usize;
    let filter_len = (filter_radius * 2 + 1).pow(2);
    let padding = (conv_layers * filter_radius - 1) as i32;
    #[allow(clippy::deref_addrof)]
    let expected = sobel_image.slice(s![padding..-padding, padding..-padding]);
    save_image(opt.output_dir.join("actual.png"), &expected.to_owned())?;
    let mut prng = Pcg64::from_seed([
        (opt.seed & 0xFF) as u8,
        ((opt.seed >> 8) & 0xFF) as u8,
        ((opt.seed >> 16) & 0xFF) as u8,
        ((opt.seed >> 24) & 0xFF) as u8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]);
    let mut random_filter = |mean: f32, variance: f32| -> Conv2<OwnedRepr<f32>> {
        // Xavier initialize by changing the variance to be 1/N where N is the number of neurons.
        Conv2::new(
            Array::from_iter(
                Normal::new(mean / filter_len as f32, variance / filter_len as f32)
                    .unwrap()
                    .sample_iter(&mut prng)
                    .take(filter_len),
            )
            .into_shape((filter_radius * 2 + 1, filter_radius * 2 + 1))
            .unwrap(),
        )
    };
    let mut prng = Pcg64::from_seed([
        (opt.seed & 0xFF) as u8,
        ((opt.seed >> 8) & 0xFF) as u8,
        ((opt.seed >> 16) & 0xFF) as u8,
        ((opt.seed >> 24) & 0xFF) as u8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]);
    let mut random_blu = |mean: f32, variance: f32| -> Blu {
        // Xavier initialize by changing the variance to be 1/N where N is the number of neurons.
        let distr = Normal::new(mean, variance).unwrap();
        Blu::new(distr.sample(&mut prng), distr.sample(&mut prng))
    };
    let mut generate_filter = || {
        random_filter(0.0, 10.0f32.powi(0))
            .chain(Map2One(random_blu(0.0, 0.05)))
            .chain(random_filter(0.0, 10.0f32.powi(0)))
            .chain(Map2One(random_blu(0.0, 0.05)))
            .chain(random_filter(0.0, 10.0f32.powi(0)))
            .chain(Map2One(random_blu(0.0, 0.05)))
            .chain(random_filter(0.0, 10.0f32.powi(0)))
            .chain(Map2One(random_blu(0.0, 0.05)))
            .chain(random_filter(64.0, 10.0f32.powi(3)))
    };

    loop {
        let mut train_filter = generate_filter();
        let mut learn_rate = opt.initial_learning_rate;
        // Weird hack to initialize the momentum to zero without knowing the shape of the ndarray in advance.
        let mut momentum = {
            let (internal, output) = train_filter.forward(&image);
            train_filter.backward_train(&image, &internal, &output)
        };
        momentum *= 0.0f32;
        for i in 0..opt.epochs {
            // Compute beta * momentum.
            momentum *= opt.beta;
            train_filter.train(&momentum);
            let (internal, output) = train_filter.forward(&image);
            // Show the image if the frame is divisible by show_every.
            if i % opt.show_every == 0 {
                save_image(opt.output_dir.join(format!("epoch{:04}.png", i)), &output)?;
            }
            // Compute the loss for display only (we don't actually need the loss itself for backprop, just its derivative).
            let loss = (output.clone() - expected.view()).map(|n| n.powi(2)).sum();
            eprintln!("epoch {:04} loss: {}, learn_rate: {}", i, loss, learn_rate);
            if i > 15 && loss > 80000.0 {
                eprintln!("loss > 80000.0 at epoch {}; starting over", i);
                break;
            }
            if i > 50 && loss > 70000.0 {
                eprintln!("loss > 70000.0 at epoch {}; starting over", i);
                break;
            }
            if i > 150 && loss > 60000.0 {
                eprintln!("loss > 60000.0 at epoch {}; starting over", i);
                break;
            }
            if i > 500 && loss > 30000.0 {
                eprintln!("loss > 60000.0 at epoch {}; starting over", i);
                break;
            }
            if !loss.is_normal() {
                eprintln!("abnormal loss at epoch {}; starting over", i);
                break;
            }
            // The loss function is (n - t)^2, so 2*(n - t) is dE/df where
            // `E` is loss and `f` is output.
            let delta_loss = (output - expected).map(|n| 2.0 * n);
            // Compute the output delta.
            let output_delta = -learn_rate * delta_loss;
            // Compute the trainable variable delta.
            let mut train_delta = train_filter.backward_train(&image, &internal, &output_delta);
            // Make the train delta a small component.
            train_delta *= 1.0 - opt.beta;
            train_filter.train(&train_delta);
            // Add the small component to the momentum.
            momentum += train_delta;
            learn_rate *= opt.learning_rate_multiplier;
        }
    }
}
