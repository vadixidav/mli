use crate::{Backward, ChainData, Forward, Train};

#[derive(Clone, Debug)]
pub struct Zip<T, U>(pub T, pub U);

impl<T, U> Forward for Zip<T, U>
where
    T: Forward,
    U: Forward,
{
    type Input = (T::Input, U::Input);
    type Internal = (T::Internal, U::Internal);
    type Output = (T::Output, U::Output);

    fn forward(&self, input: &Self::Input) -> (Self::Internal, Self::Output) {
        let (t_internal, t_output) = self.0.forward(&input.0);
        let (u_internal, u_output) = self.1.forward(&input.1);
        ((t_internal, u_internal), (t_output, u_output))
    }
}

impl<T, U> Backward for Zip<T, U>
where
    T: Backward + Forward,
    U: Backward + Forward,
{
    type OutputDelta = (T::OutputDelta, U::OutputDelta);
    type InputDelta = (T::InputDelta, U::InputDelta);
    type TrainDelta = ChainData<T::TrainDelta, U::TrainDelta>;

    fn backward(
        &self,
        input: &Self::Input,
        internal: &Self::Internal,
        output_delta: &Self::OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let (t_internal, u_internal) = internal;
        let (t_input_delta, t_train_delta) = self.0.backward(&input.0, t_internal, &output_delta.0);
        let (u_input_delta, u_train_delta) = self.1.backward(&input.1, u_internal, &output_delta.1);
        (
            (t_input_delta, u_input_delta),
            ChainData(t_train_delta, u_train_delta),
        )
    }
}

impl<T, U, O> Train for Zip<T, U>
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
