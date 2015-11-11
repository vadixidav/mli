extern crate rand;
pub mod mep;
pub use mep::Mep;
use rand::Rng;

pub trait Learning<R, In, Out> where R: Rng {
    //The mutate function performs a unit mutation. A single, several, or no actual mutations may occour.
    fn mutate(&mut self, rng: &mut R);
    //Push data through the algorithm to get outputs
    fn compute(&self, inputs: &[In], outputs: usize) -> Box<Iterator<Item=Out>>;
    //TODO: Train function
}

/*
Genetic is a trait that allows genetic manipulation. Genetic algorithms require duplication, which is why
there is a requirement for Clone. This is a more specific type of learning algorithm, so Learning is required.

It is parameterized with an Rng (R) type so that the algorithms can extract random data in the interface.

Note: This API is highly likely to change until version 1.0.
*/
pub trait Genetic<R, In, Out> : Clone + Learning<R, In, Out> where R: Rng {
    //The mate function takes a tuple of two parent references and returns a new child; this can be non-deterministic.
    fn mate(parents: (&Self, &Self), rng: &mut R) -> Self;
}
/*
/*
FunctionalAlgorithm is a trait that allows users to execute an algorithm.

It takes the normal Ins parameter for the algorithm, and also takes Param, which allows the algorithm to pass two
parameters to the provided closure of execute along with an instruction, letting the function determine the result,
which is type Ret.

A neural network might use the processor closure to take input params in one format and normalize them for internal use
as scalar values to be crunched by the network, which will spit the result out as Ret values.
*/
pub trait FunctionalAlgorithm<Ins, R, In, Param, Ret, Out, I, F1, F2>
    where R: Rng, I: Iterator<Item=Out>, F1: Fn(&Ins, Param, Param) -> Ret, F2: Fn(&mut Ins) {
    //The mate function takes a tuple of two parent references and returns a new child; this can be non-deterministic.
    fn mate(parents: (&Self, &Self), rng: &mut R) -> Self;
    //The mutate function performs a unit mutation. A single, several, or no actual mutations may occour.
    fn mutate(&mut self, rng: &mut R, mutator: F2);
    //The execute function generates an iterator to get outputs from computing the output of the genetic algorithm
    fn execute(&self, inputs: &[In], outputs: usize, processor: F1) -> I;
}

impl<Ins, R, In, Param, Ret, Out, I, F1, F2> GeneticAlgorithm<R, In, Out> for T
    where T: FunctionalAlgorithm<Ins, R, In, Param, Ret, Out, I, F1, F2>, R: Rng, I: Iterator<Item=Out>,
    F1: Fn(&Ins, Param, Param) -> Ret, F2: Fn(&mut Ins)
{
    fn mate(parents: (&Self, &Self), rng: &mut R) -> Self {
        FunctionalAlgorithm::mate(parents, rng)
    }

    fn mutate(&mut self, rng: &mut R) {

    }
}
*/
