extern crate rand;

pub mod mep;
pub use mep::*;

///SISO is an algorithm that takes a static amount of inputs and produces a static number of outputs.
pub trait SISO<'a, In, Out> {
    ///The iterator type returned from compute. Its lifetime is tied to self.
    type Iter: Iterator<Item=Out> + 'a;
    ///Compute is provided an input slice that must be the correct number of inputs and produces an
    ///iterator that provides the static number of outputs.
    fn compute(&'a self, inputs: &'a [In]) -> Self::Iter;
}

///SIVO is an algorithm that takes a static amount of inputs and produces a variable number of outputs.
pub trait SIVO<'a, In, Out> {
    ///The iterator type returned from compute. Its lifetime is tied to self.
    type Iter: Iterator<Item=Out> + 'a;
    ///Compute is provided an input slice that must be the correct number of inputs along with the number of outputs.
    ///It returns an iterator that provides the required number of outputs.
    fn compute(&'a self, inputs: &'a [In], n_outputs: usize) -> Self::Iter;
}

///VISO is an algorithm that takes a variable amount of inputs and produces a static number of outputs.
pub trait VISO<'a, In, Out> {
    ///The iterator type returned from compute. Its lifetime is tied to self.
    type Iter: Iterator<Item=Out> + 'a;
    ///Compute is provided an input slice that can be any number of inputs and produces an
    ///iterator that provides the static number of outputs.
    fn compute(&'a self, inputs: &'a [In]) -> Self::Iter;
}

///VIVO is an algorithm that takes a variable amount of inputs and produces a variable number of outputs.
pub trait VIVO<'a, In, Out> {
    ///The iterator type returned from compute. Its lifetime is tied to self.
    type Iter: Iterator<Item=Out> + 'a;
    ///Compute is provided an input slice that can be any number of inputs along with the number of outputs.
    ///It returns an iterator that provides the required number of outputs.
    fn compute(&'a self, inputs: &'a [In], n_outputs: usize) -> Self::Iter;
}

///A learning algorithm is one that can be trained and performs computations.
pub trait Learning<R, In, Out> {
    ///Train requires inputs and outputs required by the system and an Rng to introduce randomness.
    ///Level should scale linearly with the amount of time spent training and level 1 should represent the smallest
    ///amount of training possible.
    fn train(&mut self, level: u32, inputs: &[In], outputs: &[Out], rng: &mut R);
}

///Genetic is a trait that allows genetic manipulation. Genetic algorithms require duplication, which is why
///there is a requirement for Clone. It is parameterized with an Rng (R) type so that the algorithms can extract random
///data when mating so that mating can happen randomly. In and Out are parameters specifying the inputs and outputs of
///the learning algorithm component of the genetic algorithm.
///
///Note: This API is highly likely to change before version 1.0.
pub trait Genetic<R> : Clone {
    ///The mate function takes a tuple of two parent references and an Rng, then returns a new child.
    fn mate(parents: (&Self, &Self), rng: &mut R) -> Self;
    ///The mutate function performs a unit mutation. A single, several, or no actual mutations may occour.
    fn mutate(&mut self, rng: &mut R);
}
