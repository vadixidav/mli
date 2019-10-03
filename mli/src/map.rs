use crate::{Backward, ChainData, Forward, Train};

#[derive(Clone, Debug)]
pub struct Map<T, U>(pub T, pub U);

impl<T, U> Forward for Map<T, U>
where
    T: Forward,
    U: Forward<Input = T::Output>,
{
    type Input = T::Input;
    type Internal = (T::Internal, T::Output, U::Internal);
    type Output = U::Output;

    fn forward(&self, input: &T::Input) -> (Self::Internal, Self::Output) {
        let (t_internal, t_output) = self.0.forward(input);
        let (u_internal, u_output) = self.1.forward(&t_output);
        ((t_internal, t_output, u_internal), u_output)
    }
}

impl<T, U, O> Backward for Map<T, U>
where
    T: Backward<OutputDelta = U::InputDelta> + Forward<Output = O>,
    U: Backward + Forward<Input = O>,
{
    type OutputDelta = U::OutputDelta;
    type InputDelta = T::InputDelta;
    type TrainDelta = ChainData<T::TrainDelta, U::TrainDelta>;

    fn backward(
        &self,
        input: &T::Input,
        internal: &Self::Internal,
        output_delta: &U::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let (t_internal, t_output, u_internal) = internal;
        let (u_input_delta, u_train_delta) = self.1.backward(t_output, u_internal, output_delta);
        let (t_input_delta, t_train_delta) = self.0.backward(input, t_internal, &u_input_delta);
        (t_input_delta, ChainData(t_train_delta, u_train_delta))
    }
}

impl<T, U, O> Train for Map<T, U>
where
    T: Train + Backward<OutputDelta = U::InputDelta> + Forward<Output = O>,
    U: Train + Backward + Forward<Input = O>,
{
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        let ChainData(t_train_delta, u_train_delta) = train_delta;
        self.0.train(t_train_delta);
        self.1.train(u_train_delta);
    }
}
