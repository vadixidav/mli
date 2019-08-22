use crate::{Forward, Backward, Train};

#[derive(Clone, Debug)]
pub struct Chain<T, U>(pub T, pub U);

impl<T, U, Input> Forward<Input> for Chain<T, U> where T: Forward<Input>, U: Forward<T::Output> {
    type Internal = (T::Internal, T::Output, U::Internal);
    type Output = U::Output;

    fn forward(&self, input: &Input) -> (Self::Internal, Self::Output) {
        let (t_internal, t_output) = self.0.forward(input);
        let (u_internal, u_output) = self.1.forward(&t_output);
        ((t_internal, t_output, u_internal), u_output)
    }
}

impl<T, U, N, Input, OutputDelta> Backward<Input, OutputDelta> for Chain<T, U> where T: Backward<Input, U::InputDelta, Output = N>, U: Backward<N, OutputDelta> {
    type InputDelta = T::InputDelta;
    type TrainDelta = (T::TrainDelta, U::TrainDelta);

    fn backward(
        &self,
        input: &Input,
        internal: &Self::Internal,
        output_delta: &OutputDelta,
    ) -> (Self::InputDelta, Self::TrainDelta) {
        let (t_internal, t_output, u_internal) = internal;
        let (u_input_delta, u_train_delta) = self.1.backward(t_output, u_internal, output_delta);
        let (t_input_delta, t_train_delta) = self.0.backward(input, t_internal, &u_input_delta);
        (t_input_delta, (t_train_delta, u_train_delta))
    }
}

impl<T, U, N, Input, OutputDelta> Train<Input, OutputDelta> for Chain<T, U> where T: Train<Input, U::InputDelta, Output = N>, U: Train<N, OutputDelta> {
    fn train(&mut self, train_delta: &Self::TrainDelta) {
        let (t_train_delta, u_train_delta) = train_delta;
        self.0.train(t_train_delta);
        self.1.train(u_train_delta);
    }
}