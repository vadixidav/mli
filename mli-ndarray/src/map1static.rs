use itertools::izip;
use mli::{Backward, EmptyData, Forward, Train};
use ndarray::{Array, Array1};

#[derive(Clone, Debug)]
pub struct Map1Static<G>(pub G);

impl<G> Forward for Map1Static<G>
where
    G: Forward<Internal = EmptyData>,
{
    type Input = Array1<G::Input>;
    type Internal = EmptyData;
    type Output = Array1<G::Output>;

    fn forward(&self, input: &Self::Input) -> (EmptyData, Self::Output) {
        let output_vec: Vec<G::Output> =
            input.iter().map(|input| self.0.forward(input).1).collect();
        let output_array = Array::from_shape_vec(input.raw_dim(), output_vec).unwrap();
        (EmptyData, output_array)
    }
}

impl<G> Backward for Map1Static<G>
where
    G: Backward<TrainDelta = EmptyData> + Forward<Internal = EmptyData>,
{
    type OutputDelta = Array1<G::OutputDelta>;
    type InputDelta = Array1<G::InputDelta>;
    type TrainDelta = EmptyData;

    fn backward(
        &self,
        input: &Self::Input,
        _: &EmptyData,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, EmptyData) {
        let input_delta_vec: Vec<G::InputDelta> = izip!(input.iter(), output_delta.iter(),)
            .map(|(input, output_delta)| self.0.backward(input, &EmptyData, output_delta).0)
            .collect();
        let input_delta_array = Array::from_shape_vec(input.raw_dim(), input_delta_vec).unwrap();
        (input_delta_array, EmptyData)
    }
}

impl<G> Train for Map1Static<G>
where
    G: Backward<TrainDelta = EmptyData> + Forward<Internal = EmptyData>,
{
    #[inline]
    fn train(&mut self, _: &EmptyData) {}
}
