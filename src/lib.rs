extern crate rand;
pub mod mep;
pub use mep::Mep;
use rand::Rng;

/*
A learning algorithm is one that can be trained and performs computations.
*/
pub trait Learning<R, In, Out> where R: Rng {
    //Push data through the algorithm to get outputs
    fn compute<'a>(&'a self, inputs: &'a [In], outputs: usize) -> Box<Iterator<Item=Out> + 'a>;
    //Train requires inputs and outputs required by the system and an Rng to introduce randomness
    fn train(&self, inputs: &[In], outputs: &[Out], rng: &mut R);
}

/*
Genetic is a trait that allows genetic manipulation. Genetic algorithms require duplication, which is why
there is a requirement for Clone. This is a more specific type of learning algorithm, so Learning is required.

It is parameterized with an Rng (R) type so that the algorithms can extract random data in the interface.

In and Out are parameters specifying the inputs and outputs of the learning algorithm component of the genetic
algorithm.

Note: This API is highly likely to change until version 1.0.
*/
pub trait Genetic<R, In, Out> : Clone + Learning<R, In, Out> where R: Rng {
    //The mate function takes a tuple of two parent references and returns a new child; this can be non-deterministic.
    fn mate(parents: (&Self, &Self), rng: &mut R) -> Self;
    //The mutate function performs a unit mutation. A single, several, or no actual mutations may occour.
    fn mutate(&mut self, rng: &mut R);
}
