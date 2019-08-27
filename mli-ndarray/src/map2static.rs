use itertools::izip;
use mli::{Backward, Forward, Train};
use ndarray::{Array, Array2};

#[derive(Clone, Debug)]
pub struct Map2Static<G>(pub G);

impl<G, Input> Forward<Array2<Input>> for Map2Static<G>
where
    G: Forward<Input, Internal = ()>,
{
    type Internal = ();
    type Output = Array2<G::Output>;

    fn forward(&self, input: &Array2<Input>) -> ((), Self::Output) {
        let output_vec: Vec<G::Output> =
            input.iter().map(|input| self.0.forward(input).1).collect();
        let output_array = Array::from_shape_vec(input.raw_dim(), output_vec).unwrap();
        ((), output_array)
    }
}

impl<G, Input, OutputDelta> Backward<Array2<Input>, Array2<OutputDelta>> for Map2Static<G>
where
    G: Backward<Input, OutputDelta, TrainDelta = ()> + Forward<Input, Internal = ()>,
{
    type InputDelta = Array2<G::InputDelta>;
    type TrainDelta = ();

    fn backward(
        &self,
        input: &Array2<Input>,
        _: &(),
        output_delta: &Array2<OutputDelta>,
    ) -> (Self::InputDelta, ()) {
        let input_delta_vec: Vec<G::InputDelta> = izip!(input.iter(), output_delta.iter(),)
            .map(|(input, output_delta)| self.0.backward(input, &(), output_delta).0)
            .collect();
        let input_delta_array = Array::from_shape_vec(input.raw_dim(), input_delta_vec).unwrap();
        (input_delta_array, ())
    }
}

impl<G, Input, OutputDelta> Train<Array2<Input>, Array2<OutputDelta>> for Map2Static<G>
where
    G: Backward<Input, OutputDelta, TrainDelta = ()> + Forward<Input, Internal = ()>,
{
    #[inline]
    fn train(&mut self, _: &()) {}
}
