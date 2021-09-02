use crate::Ndeep;
use itertools::izip;
use mli::{Backward, Forward, Train};
use ndarray::{azip, Array, Array3, Ix3, OwnedRepr};

#[derive(Clone, Debug)]
pub struct Map3Many<G>(pub Array3<G>);

impl<G> Forward for Map3Many<G>
where
    G: Forward,
{
    type Input = Array3<G::Input>;
    type Internal = Array3<G::Internal>;
    type Output = Array3<G::Output>;

    fn forward(&self, input: &Self::Input) -> (Self::Internal, Self::Output) {
        assert_eq!(input.shape(), self.0.shape());
        let both = input
            .iter()
            .zip(self.0.iter())
            .map(|(input, op)| op.forward(input));
        let (internal_vec, output_vec) = both.fold(
            (vec![], vec![]),
            |(mut internal_vec, mut output_vec), (internal, output)| {
                internal_vec.push(internal);
                output_vec.push(output);
                (internal_vec, output_vec)
            },
        );
        let internal_array = Array::from_shape_vec(input.raw_dim(), internal_vec).unwrap();
        let output_array = Array::from_shape_vec(input.raw_dim(), output_vec).unwrap();
        (internal_array, output_array)
    }
}

impl<G> Backward for Map3Many<G>
where
    G: Backward,
{
    type OutputDelta = Array3<G::OutputDelta>;
    type InputDelta = Array3<G::InputDelta>;
    type TrainDelta = Ndeep<OwnedRepr<G::TrainDelta>, Ix3>;

    fn backward(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        assert_eq!(input.shape(), self.0.shape());
        let both = izip!(
            input.iter(),
            internal.iter(),
            output_delta.iter(),
            self.0.iter()
        )
        .map(|(input, internal, output_delta, op)| op.backward(input, internal, output_delta));
        let (input_delta_vec, train_delta_vec) = both.fold(
            (vec![], vec![]),
            |(mut input_delta_vec, mut train_delta_vec), (input_delta, train_delta)| {
                input_delta_vec.push(input_delta);
                train_delta_vec.push(train_delta);
                (input_delta_vec, train_delta_vec)
            },
        );
        let input_delta_array = Array::from_shape_vec(input.raw_dim(), input_delta_vec).unwrap();
        let train_delta_array = Array::from_shape_vec(input.raw_dim(), train_delta_vec).unwrap();
        (input_delta_array, Ndeep(train_delta_array))
    }
}

impl<G> Train for Map3Many<G>
where
    G: Train,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        let ops = self.0.view_mut();
        let train_delta = train_delta.0.view();
        azip!((ops in ops, train_delta in train_delta) {
            ops.train(train_delta);
        });
    }
}
