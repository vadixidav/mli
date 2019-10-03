use image::ImageResult;
use mli::{Backward, Forward, Graph, Train};
use mli_conv::{Conv2, Conv2n, Conv3};
use mli_defconv::DefConv2InternalOffsets;
use mli_ndarray::{Map2One, Map3One, Reshape3to2};
use mli_relu::Blu;
use mnist::{Mnist, MnistBuilder};
use ndarray::{array, s, Array, Array2, Array3, ArrayView, ArrayView3, OwnedRepr};
use ndarray_image::{open_gray_image, save_gray_image, save_image, Colors};
use rand_core::{RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64;
use std::path::{Path, PathBuf};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "mnist", about = "An example of using MLI to classify MNIST")]
struct Opt {
    /// Number of epochs
    #[structopt(short = "t", default_value = "4000")]
    epochs: usize,
    /// Number of epochs per output image
    #[structopt(short = "s", default_value = "1")]
    show_every: usize,
    /// Pre-start learning rate (for first 10 epochs)
    #[structopt(short = "p", default_value = "0.1")]
    prestart_learning_rate: f32,
    /// Initial learning rate
    #[structopt(short = "i", default_value = "0.3")]
    initial_learning_rate: f32,
    /// Learning rate multiplier per epoch
    #[structopt(short = "m", default_value = "0.999")]
    learning_rate_multiplier: f32,
    /// Seed
    #[structopt(short = "z", default_value = "34")]
    seed: u32,
    /// Beta value for NAG
    #[structopt(short = "b", default_value = "0.9")]
    beta: f32,
    /// Directory containing the MNIST files
    #[structopt(parse(from_os_str))]
    mnist_dir: PathBuf,
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

fn save_image_internal(path: impl AsRef<Path>, image: &Array2<f32>) -> ImageResult<()> {
    let image = image.map(|&n| num::clamp(n * 255.0, 0.0, 255.0) as u8);
    save_gray_image(path, image.view())
}

fn save_image_color_internal(path: impl AsRef<Path>, image: &Array3<f32>) -> ImageResult<()> {
    let image = image.map(|&n| num::clamp(n * 255.0, 0.0, 255.0) as u8);
    save_image(path, image.view(), Colors::Rgb)
}

fn mnist_train<'a>(mnist: &'a Mnist) -> ArrayView3<'a, u8> {
    ArrayView::from_shape((60000, 28, 28), mnist.trn_img.as_slice()).expect("mnist data corrupted")
}

fn main() -> ImageResult<()> {
    let opt = Opt::from_args();
    let mut prng = Pcg64::from_seed([0; 32]);
    let mnist = MnistBuilder::new()
        .base_path(&opt.mnist_dir.display().to_string())
        .label_format_digit()
        .finalize();
    let train = mnist_train(&mnist).mapv(|u| u as f32 / 255.0);

    ////////////////////////
    // Defining the model //
    ////////////////////////

    let conv_layers = 4;
    let filter_radius = 1usize;
    let filter_depth = 2usize;
    let filter_area = (filter_radius * 2 + 1).pow(2);
    let filter_volume = filter_area * filter_depth;

    let defconv_total_strides = 28;

    let padding = (conv_layers * filter_radius - 1) as i32;
    let mut prng = make_prng(opt.seed);
    let mut prng_defconv = make_prng(prng.next_u32());
    let mut random_defconv =
        |samples: usize, mean: f32, variance: f32| -> DefConv2InternalOffsets {
            // Xavier initialize by changing the variance to be 1/N where N is the area of the filter.
            DefConv2InternalOffsets::new(
                Array::from_iter(
                    Normal::new(mean / samples as f32, variance / samples as f32)
                        .unwrap()
                        .sample_iter(&mut prng_defconv)
                        .take(samples),
                )
                .into_shape(samples)
                .unwrap(),
                Array::from_iter(
                    Normal::new(0.0, 5.0)
                        .unwrap()
                        .sample_iter(&mut prng_defconv)
                        .take(2 * samples),
                )
                .into_shape((samples, 2))
                .unwrap(),
                [defconv_total_strides, defconv_total_strides],
            )
        };
    let mut prng_2filter = make_prng(prng.next_u32());
    let mut random_2filter = |mean: f32, variance: f32| -> Conv2<OwnedRepr<f32>> {
        // Xavier initialize by changing the variance to be 1/N where N is the area of the filter.
        Conv2::new(
            Array::from_iter(
                Normal::new(mean / filter_area as f32, variance / filter_area as f32)
                    .unwrap()
                    .sample_iter(&mut prng_2filter)
                    .take(filter_area),
            )
            .into_shape((filter_radius * 2 + 1, filter_radius * 2 + 1))
            .unwrap(),
        )
    };
    let mut prng_2nfilter = make_prng(prng.next_u32());
    let mut random_2nfilter = |mean: f32, variance: f32| -> Conv2n<OwnedRepr<f32>> {
        // Xavier initialize by changing the variance to be 1/N where N is the area of the filter.
        Conv2n::new(
            Array::from_iter(
                Normal::new(mean / filter_area as f32, variance / filter_area as f32)
                    .unwrap()
                    .sample_iter(&mut prng_2nfilter)
                    .take(filter_volume),
            )
            .into_shape((filter_depth, filter_radius * 2 + 1, filter_radius * 2 + 1))
            .unwrap(),
        )
    };
    let mut prng_3filter = make_prng(prng.next_u32());
    let mut random_3filter = |mean: f32, variance: f32| -> Conv3<OwnedRepr<f32>> {
        // Xavier initialize by changing the variance to be 1/N where N is the volume of the filter.
        Conv3::new(
            Array::from_iter(
                Normal::new(mean / filter_volume as f32, variance / filter_volume as f32)
                    .unwrap()
                    .sample_iter(&mut prng_3filter)
                    .take(filter_volume),
            )
            .into_shape((filter_depth, filter_radius * 2 + 1, filter_radius * 2 + 1))
            .unwrap(),
        )
    };
    let mut prng_blu = make_prng(prng.next_u32());
    let mut random_blu = |mean: f32, variance: f32| -> Blu {
        // Xavier initialize by changing the variance to be 1/N where N is the number of neurons.
        let distr = Normal::new(mean, variance).unwrap();
        Blu::new(distr.sample(&mut prng_blu), distr.sample(&mut prng_blu))
    };
    let mut generate_filter = || {
        random_defconv(10, 1.0, 0.5)
            .map(random_2nfilter(0.0, 4.0))
            .map(Map3One(random_blu(0.0, 0.5)))
            .map(random_3filter(0.0, 4.0))
            .map(Reshape3to2::new())
            .map(Map2One(random_blu(0.0, 0.5)))
            .map(random_2filter(0.0, 4.0))
            .map(Map2One(random_blu(0.0, 0.5)))
            .map(random_2filter(0.0, 4.0))
    };

    //////////////
    // Training //
    //////////////

    loop {
        let mut train_filter = generate_filter();
        let mut learn_rate = opt.initial_learning_rate;
        // Weird hack to initialize the momentum to zero without knowing the shape of the ndarray in advance.
        let dummy_image = train.outer_iter().nth(0).unwrap().to_owned();
        let mut momentum = {
            let (internal, output) = train_filter.forward(&dummy_image);
            train_filter.backward_train(&dummy_image, &internal, &output)
        };
        let mut last_loss = 0.0;
        momentum *= 0.0f32;
        for i in 0..opt.epochs {
            // Iterate through every image in the epoch.
            for (sample_ix, (image, &label)) in train
                .outer_iter()
                .map(|a| a.to_owned())
                .zip(mnist.trn_lbl.iter())
                .enumerate()
            {
                // Compute beta * momentum.
                momentum *= opt.beta;
                train_filter.train(&momentum);
                let (internal, output) = train_filter.forward(&image);
                // Show the image if the frame is divisible by show_every.
                if sample_ix % opt.show_every == 0 {
                    // Plot the sample locations from the deformable conv net in the center.
                    let mut splat: Array3<f32> = Array::zeros([1024, 1024, 3]);
                    let defconv = &(((((((train_filter.0).0).0).0).0).0).0).0;
                    // Get the weight distance so we can normalize the weights.
                    let weight_distance = defconv
                        .def_conv
                        .weights
                        .iter()
                        .map(|&n| n.powi(2))
                        .sum::<f32>()
                        .sqrt();
                    // Draw all the offsets and weights.
                    for (offset, &weight) in defconv
                        .offsets
                        .0
                        .outer_iter()
                        .zip(defconv.def_conv.weights.iter())
                    {
                        let splat_offset = [
                            (offset[0] * 32.0 + splat.shape()[0] as f32 / 2.0).round(),
                            (offset[1] * 32.0 + splat.shape()[1] as f32 / 2.0).round(),
                        ];
                        if splat_offset[0] >= 0.0
                            && splat_offset[0] < splat.shape()[0] as f32
                            && splat_offset[1] >= 0.0
                            && splat_offset[1] < splat.shape()[1] as f32
                        {
                            splat[[splat_offset[0] as usize, splat_offset[1] as usize, 0]] =
                                weight / weight_distance;
                            splat[[splat_offset[0] as usize, splat_offset[1] as usize, 1]] =
                                1.0 - weight / weight_distance;
                        }
                    }
                    // Draw the output for the epoch.
                    save_image_internal(
                        opt.output_dir
                            .join(format!("epoch{:03}_sample{:06}.png", i, sample_ix)),
                        &output,
                    )?;
                    // Draw the splat for the epoch.
                    save_image_color_internal(
                        opt.output_dir
                            .join(format!("splat{:03}_sample{:06}.png", i, sample_ix)),
                        &splat,
                    )?;
                }
                if i == opt.epochs - 1 {
                    eprintln!("Finished!");
                    return Ok(());
                }
                let output_len = output.len() as f32;
                let local_learn_rate = if i < 10 {
                    opt.prestart_learning_rate
                } else {
                    let llr = learn_rate;
                    learn_rate *= opt.learning_rate_multiplier;
                    llr
                };
                // Compute the loss for display only (we don't actually need the loss itself for backprop, just its derivative).
                let mut delta_loss = output.clone();
                let start = delta_loss.len() / 2;
                for (ix, v) in delta_loss.iter_mut().enumerate() {
                    if ix >= start && ix < start + 10 {
                        let expected = if (ix - start) == label as usize {
                            1.0
                        } else {
                            0.0
                        };
                        *v = 2.0 * (*v - expected);
                    } else {
                        *v = 0.0;
                    }
                }
                let loss = delta_loss.iter().map(|&n| (n / 2.0).powi(2)).sum();
                eprintln!(
                    "epoch {:03} sample {:06} loss(ish): {} ({:+}%), learn_rate: {}",
                    i,
                    sample_ix,
                    loss,
                    (loss - last_loss) / last_loss * 100.0,
                    local_learn_rate
                );
                last_loss = loss;
                if !loss.is_normal() {
                    eprintln!("abnormal loss at epoch {}; starting over", i);
                    break;
                }
                // Compute the output delta.
                let output_delta = -local_learn_rate * delta_loss / output_len;
                // Compute the trainable variable delta.
                let mut train_delta = train_filter.backward_train(&image, &internal, &output_delta);
                // Make the train delta a small component.
                train_delta *= 1.0 - opt.beta;
                train_filter.train(&train_delta);
                // Add the small component to the momentum.
                momentum += train_delta;
            }
        }
    }
}

fn make_prng(seed: u32) -> Pcg64 {
    Pcg64::from_seed([
        (seed & 0xFF) as u8,
        ((seed >> 8) & 0xFF) as u8,
        ((seed >> 16) & 0xFF) as u8,
        ((seed >> 24) & 0xFF) as u8,
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
    ])
}