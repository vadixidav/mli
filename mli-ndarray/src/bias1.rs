use crate::Ndeep;
use mli::*;
use ndarray::{Array1, ArrayBase, Data, OwnedRepr};
use std::marker::PhantomData;

type D1 = ndarray::Ix1;

#[derive(Clone, Debug)]
pub struct Bias1<S>(Array1<f32>, PhantomData<S>);

impl<S> Bias1<S> {
    /// The dimensions of the filters array are `[filter, col]`.
    pub fn new(biases: Array1<f32>) -> Self {
        Self(biases, PhantomData)
    }
}

impl<S> Forward for Bias1<S>
where
    S: Data<Elem = f32>,
{
    type Input = ArrayBase<S, D1>;
    type Internal = ();
    type Output = Array1<f32>;

    fn forward(&self, input: &Self::Input) -> ((), Self::Output) {
        let Self(biases, _) = self;
        assert_eq!(
            input.shape(),
            biases.shape(),
            "input shape does not match biases"
        );

        ((), biases.clone() + input)
    }
}

impl<S> Backward for Bias1<S>
where
    S: Data<Elem = f32>,
{
    type OutputDelta = Array1<f32>;
    type InputDelta = Array1<f32>;
    type TrainDelta = Ndeep<OwnedRepr<f32>, D1>;

    fn backward(
        &self,
        _: &Self::Input,
        _: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        (output_delta.clone(), Ndeep(output_delta.clone()))
    }
}

impl<S> Train for Bias1<S>
where
    S: Data<Elem = f32>,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        self.0 += &train_delta.0;
    }
}
