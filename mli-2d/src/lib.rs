use ndarray::{Array2, Array, azip};
use mli::{Forward, Backward, Train};

#[derive(Clone, Debug)]
pub struct Map2Many<G>(pub Array2<G>);

impl<G, Input> Forward<Array2<Input>> for Map2Many<G>
    where G: Forward<Input>
{
    type Internal = Array2<G::Internal>;
    type Output = Array2<G::Output>;

    fn forward(&self, input: &Array2<Input>) -> (Self::Internal, Self::Output) {
        assert_eq!(input.shape(), self.0.shape());
        let both_vec: Vec<(G::Internal, G::Output)> = input.iter()
                .zip(self.0.iter())
                .map(|(input, op)| op.forward(input))
                .collect();
        let (internal_vec, output_vec) = both_vec.into_iter().fold((vec![], vec![]), |(mut internal_vec, mut output_vec), (internal, output)| {
            internal_vec.push(internal);
            output_vec.push(output);
            (internal_vec, output_vec)
        });
        let internal_array = Array::from_shape_vec(
            input.raw_dim(),
            internal_vec,
        ).unwrap();
        let output_array = Array::from_shape_vec(
            input.raw_dim(),
            output_vec,
        ).unwrap();
        (internal_array, output_array)
    }
}