use image::open;
use mli::Forward;
use mli_conv::Conv2;
use ndarray::{array, Array};
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
    let (width, height) = image.dimensions();
    let dims = (height as usize, width as usize);
    let input_image = Array::from_shape_vec(dims, image.into_raw()).unwrap();
    let input_image = input_image.map(|&n| n as f32);
    let filter = array![[-2.0, -2.0, 0.0], [-2.0, 0.0, 2.0], [0.0, 2.0, 2.0],];
    let conv = Conv2(filter);
    let output_image = conv.forward(input_image);
    let output_image = output_image.map(|&n| num::clamp(n, 0.0, 255.0) as u8);
    if let &[height, width] = output_image.shape() {
        let vec = output_image.into_raw_vec();
        let image = image::GrayImage::from_raw(width as u32, height as u32, vec)
            .expect("failed to create image from raw vec");
        image.save(opt.output).expect("failed to save file");
    } else {
        panic!("the output image had more than 2 dimensions");
    }
}
