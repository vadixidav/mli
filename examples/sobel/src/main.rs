use mli::Forward;
use mli_conv::Conv2;
use ndarray::array;
use ndarray_image::{open_gray_image, save_gray_image};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "image", about = "Loads and convolves an image as a test")]
struct Opt {
    /// File to load
    #[structopt(parse(from_os_str))]
    file: PathBuf,
    /// Output location
    #[structopt(parse(from_os_str))]
    output: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    let image = open_gray_image(opt.file).expect("unable to open input image");
    let image = image.map(|&n| n.into());
    let down_filter = array![[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0],];
    let right_filter = down_filter.t().to_owned();
    let down = Conv2::new(down_filter).forward(&image).1;
    let right = Conv2::new(right_filter).forward(&image).1;
    let image = (down.map(|f| f.powi(2)) + right.map(|f| f.powi(2))).map(|f| f.sqrt());
    let image = image.map(|&n| num::clamp(n, 0.0, 255.0) as u8);
    save_gray_image(opt.output, image.view()).expect("failed to write output");
}
