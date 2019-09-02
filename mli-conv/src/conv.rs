use ndarray::{Array, Array2, Array3, ArrayView2, ArrayView3};

pub fn convolve2<'a>(signal: ArrayView2<'a, f32>, filter: ArrayView2<'a, f32>) -> Array2<f32> {
    let filter_dims = filter.raw_dim();
    let output_dims = (
        signal.shape()[0] + 1 - filter_dims[0],
        signal.shape()[1] + 1 - filter_dims[1],
    );
    Array::from_shape_vec(
        output_dims,
        signal
            .windows(filter_dims)
            .into_iter()
            .map(|view| (view.to_owned() * filter).sum())
            .collect(),
    )
    .expect("2d convolution produced incorrectly sized output")
}

pub fn convolve3<'a>(signal: ArrayView3<'a, f32>, filter: ArrayView3<'a, f32>) -> Array3<f32> {
    let filter_dims = filter.raw_dim();
    let output_dims = (
        signal.shape()[0] + 1 - filter_dims[0],
        signal.shape()[1] + 1 - filter_dims[1],
        signal.shape()[2] + 1 - filter_dims[2],
    );
    Array::from_shape_vec(
        output_dims,
        signal
            .windows(filter_dims)
            .into_iter()
            .map(|view| (view.to_owned() * filter).sum())
            .collect(),
    )
    .expect("3d convolution produced incorrectly sized output")
}

pub fn convolve2n<'a>(signal: ArrayView2<'a, f32>, filters: ArrayView3<'a, f32>) -> Array3<f32> {
    let filter_dims = (filters.shape()[1], filters.shape()[2]);
    let output_dims = (
        filters.shape()[0],
        signal.shape()[0] + 1 - filter_dims.0,
        signal.shape()[1] + 1 - filter_dims.1,
    );
    Array::from_shape_vec(
        output_dims,
        filters
            .outer_iter()
            .flat_map(|filter| {
                signal
                    .windows(filter_dims)
                    .into_iter()
                    .map(move |view| (view.to_owned() * filter).sum())
            })
            .collect(),
    )
    .expect("convolution produced incorrectly sized output")
}
