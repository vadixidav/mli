use itertools::izip;
use mli::{Backward, Forward, Train};
use ndarray::{Array, Array2};

#[derive(Clone, Debug)]
pub struct Map2Static<G>(pub G);

impl<G> Forward for Map2Static<G>
where
    G: Forward<Internal = ()>,
{
    type Input = Array2<G::Input>;
    type Internal = ();
    type Output = Array2<G::Output>;

    fn forward(&self, input: &Self::Input) -> ((), Self::Output) {
        let output_vec: Vec<G::Output> =
            input.iter().map(|input| self.0.forward(input).1).collect();
        let output_array = Array::from_shape_vec(input.raw_dim(), output_vec).unwrap();
        ((), output_array)
    }
}

impl<G> Backward for Map2Static<G>
where
    G: Backward<TrainDelta = ()> + Forward<Internal = ()>,
{
    type OutputDelta = Array2<G::OutputDelta>;
    type InputDelta = Array2<G::InputDelta>;
    type TrainDelta = ();

    fn backward(
        &self,
        input: &Self::Input,
        _: &(),
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, ()) {
        let input_delta_vec: Vec<G::InputDelta> = izip!(input.iter(), output_delta.iter(),)
            .map(|(input, output_delta)| self.0.backward(input, &(), output_delta).0)
            .collect();
        let input_delta_array = Array::from_shape_vec(input.raw_dim(), input_delta_vec).unwrap();
        (input_delta_array, ())
    }
}

impl<G> Train for Map2Static<G>
where
    G: Backward<TrainDelta = ()> + Forward<Internal = ()>,
{
    #[inline]
    fn train(&mut self, _: &()) {}
}
