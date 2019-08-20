use image::open;
use mli::Forward;
use mli_conv::Conv2;
use ndarray::array;
use ndarray_image::{ImgLuma, NdGray, NdImage};
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
    let image = open(opt.file).expect("unable to open input image");
    let image = image.to_luma();
    let image: NdGray = NdImage(&image).into();
    let image = image.map(|&n| n as f32);
    let down_filter = array![[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0],];
    let right_filter = down_filter.t().to_owned();
    fn square(f: &f32) -> f32 {
        f.powi(2)
    }
    fn sqrt(f: &f32) -> f32 {
        f.sqrt()
    }
    let down = Conv2(down_filter).forward(&image.view()).1;
    let right = Conv2(right_filter).forward(&image.view()).1;
    let image = (down.map(square) + right.map(square)).map(sqrt);
    let image = image.map(|&n| num::clamp(n, 0.0, 255.0) as u8);
    let image: Option<ImgLuma> = NdImage(image.view()).into();
    let image = image.expect("failed to convert ndarray to image");
    image.save(opt.output).expect("failed to write output");
}
