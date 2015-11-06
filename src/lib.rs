extern crate rand;
pub mod mep;
pub use mep::Mep;
use rand::Rng;

/*
GeneticAlgorithm is a trait that allows genetic manipulation. Genetic algorithms require duplication, which is why
there is a requirement for Clone.

It is parameterized with an Rng (R) type so that the algorithms can extract random data in the interface.

It is also parameterized with an instruction type (Ins). This allows a mutator closure to modify the instruction.

The C container parameter specifies the immutable reference type that call is passed to perform some action.

Note: This API is highly likely to change until version 1.0.
*/
pub trait GeneticAlgorithm<R, Ins, C> : Clone where R: Rng {
    //The mate function takes a tuple of two parent references and returns a new child; this can be non-deterministic.
    fn mate(parents: (&Self, &Self), rng: &mut R) -> Self;
    //The mutate function performs a unit mutation. A single, several, or no actual mutations may occour.
    fn mutate<F>(&mut self, rng: &mut R, mutator: F) where F: FnMut(&mut Ins);
    //The call function allows a closure to aquire a program so it can operate on it.
    fn call<F>(&self, program: F) where F: FnOnce(&C);
}
