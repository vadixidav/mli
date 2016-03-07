extern crate num;
use std::collections::BTreeSet;
use std::cmp;
use rand::Rng;
use std::ops::Range;
use std::iter::Rev;
use super::{Genetic, SISO};
use std::marker::PhantomData;

///Defines an opcode for the Mep. Every opcode contains an instruction and two parameter indices. These specify which
///previous opcodes produced the result required as inputs to this opcode. These parameters can also come from the inputs
///to the program, which sequentially preceed the internal instructions.
#[derive(Clone, Copy)]
struct Opcode<Ins> {
    instruction: Ins,
    first: usize,
    second: usize,
}

///A multi-expression program represented using a series of operations that can reuse results of previous operations.
pub struct Mep<Ins, R, Param, F1, F2> {
    program: Vec<Opcode<Ins>>,
    pub unit_mutate_size: usize,
    pub crossover_points: usize,
    inputs: usize,
    outputs: usize,
    mutator: F1,
    processor: F2,
    _phantom: (PhantomData<R>, PhantomData<Param>),
}

impl<Ins, R, Param, F1, F2> Clone for Mep<Ins, R, Param, F1, F2>
    where Ins: Clone, F1: Copy, F2: Copy
{
    fn clone(&self) -> Self {
        Mep{
            program: self.program.clone(),
            unit_mutate_size: self.unit_mutate_size,
            crossover_points: self.crossover_points,
            inputs: self.inputs,
            outputs: self.outputs,
            mutator: self.mutator,
            processor: self.processor,
            _phantom: (PhantomData, PhantomData),
        }
    }
}

impl<Ins, R, Param, F1, F2> Mep<Ins, R, Param, F1, F2> {
    ///Generates a new Mep with a particular size and takes a closure to generate random instructions.
    ///Takes an RNG as well to generate random internal data for each instruction. This Rng is different from the Mep's
    ///Rng, thus it is parameterized separately here.
    pub fn new<I, Rg>(inputs: usize, outputs: usize, unit_mutate_size: usize, crossover_points: usize, rng: &mut Rg,
        instruction_iter: I, mutator: F1, processor: F2) -> Self
        where I: Iterator<Item=Ins>, Rg: Rng, F1: Fn(&mut Ins, &mut R), F2: Fn(&Ins, Param, Param) -> Param
    {
        Mep{
            program: instruction_iter.enumerate()
                .map(|(index, ins)| Opcode{
                        instruction: ins,
                        first: rng.gen_range(0, index + inputs),
                        second: rng.gen_range(0, index + inputs),
                    }
                ).collect(),
            unit_mutate_size: unit_mutate_size,
            crossover_points: crossover_points,
            inputs: inputs,
            outputs: outputs,
            mutator: mutator,
            processor: processor,
            _phantom: (PhantomData, PhantomData),
        }
    }
}

impl<Ins, R, Param, F1, F2> Genetic<R> for Mep<Ins, R, Param, F1, F2>
    where R: Rng, Ins: Clone, Param: Clone, F1: Copy + Fn(&mut Ins, &mut R), F2: Copy
{
    fn mate(parents: (&Self, &Self), rng: &mut R) -> Self {
        //Each Mep must have the same amount of inputs
        //TODO: Once Rust implements generic values, this can be made explicit and is not needed
        assert!(parents.0.inputs == parents.1.inputs);
        assert!(parents.0.outputs == parents.1.outputs);
        //Get the smallest of the two lengths
        let total_instructions = cmp::min(parents.0.program.len(), parents.1.program.len());
        let crossover_choice = rng.gen_range(0, 2);
        Mep{program:
            //Generate a randomly sized sequence between 1 and crossover_points
            (0..rng.gen_range(1, cmp::min(total_instructions / 2, {
                if crossover_choice == 0 {parents.0}
                else {parents.1}}.crossover_points)))
            //Map these to random crossover points
            .map(|_| rng.gen_range(0, total_instructions))
            //Add total_instructions at the end so we can generate a range with it
            .chain(Some(total_instructions))
            //Sort them by value into BTree, which removes duplicates and orders them
            .fold(BTreeSet::new(), |mut set, i| {set.insert(i); set})
            //Iterate over the sorted values
            .iter()
            //Turn every copy of two, prepending a 0, into a range
            .scan(0, |prev, x| {let out = Some(*prev..*x); *prev = *x; out})
            //Enumerate by index to differentiate odd and even values
            .enumerate()
            //Map even pairs to ranges in parent 0 and odd ones to ranges in parent 1 and expand the ranges
            .flat_map(|(index, range)| {
                {if index % 2 == 0 {parents.0} else {parents.1}}.program[range].iter().cloned()
            })
            //Collect all the instruction ranges from each parent
            .collect(),

            unit_mutate_size: if parents.0.unit_mutate_size < parents.1.unit_mutate_size {
                rng.gen_range(parents.0.unit_mutate_size, parents.1.unit_mutate_size + 1)
            } else {
                rng.gen_range(parents.1.unit_mutate_size, parents.0.unit_mutate_size + 1)
            },

            crossover_points: if parents.0.crossover_points < parents.1.crossover_points {
                rng.gen_range(parents.0.crossover_points, parents.1.crossover_points + 1)
            } else {
                rng.gen_range(parents.1.crossover_points, parents.0.crossover_points + 1)
            },

            inputs: parents.0.inputs,
            outputs: parents.0.outputs,
            mutator: {if rng.gen_range(0, 2) == 0 {parents.0} else {parents.1}}.mutator,
            processor: {if rng.gen_range(0, 2) == 0 {parents.0} else {parents.1}}.processor,
            _phantom: (PhantomData, PhantomData),
        }
    }

    ///The Mep mutate function operates using the unit_mutate_size. This variable specifies the amount of instructions
    ///for which a single mutation is expect to occour every time mutate is called. This variable can be mutated inside
    ///of mutate, in which case it may never go below 1, but may tend towards infinity in increments of 1. This variable
    ///is implemented as a u64 to permit it to expand unbounded to mutation levels that are so low that mutations
    ///virtually never happen. Allowing this to mutate allows species to find the equilibrium between genomic
    ///adaptability and stability. If a species develops information gathering, then it can adapt intellegently, making
    ///it possibly more beneficial to operate at lower mutation rates. Setting the default mutation rate for species
    ///properly, or allowing it to adapt as the simulation continues, permits species to survive more frequently that
    ///are randomly generated.
    fn mutate(&mut self, rng: &mut R) {
        //Mutate unit_mutate_size
        if rng.gen_range(0, self.unit_mutate_size) == 0 {
            //Make it possibly go up or down by 1
            match rng.gen_range(0, 2) {
                0 => self.unit_mutate_size += 1,
                1 => if self.unit_mutate_size > 1 {self.unit_mutate_size -= 1},
                _ => unreachable!(),
            }
        }
        //Mutate crossover_points
        if rng.gen_range(0, self.unit_mutate_size) == 0 {
            //Make it possibly go up or down by 1
            match rng.gen_range(0, 2) {
                0 => self.crossover_points += 1,
                1 => if self.crossover_points > 1 {self.crossover_points -= 1},
                _ => unreachable!(),
            }
        }

        //Mutate the instructions using the mutator
        loop {
            //Choose a random location in the instructions and then add a random value up to the unit_mutate_size
            let choice = rng.gen_range(0, self.program.len()) + rng.gen_range(0, self.unit_mutate_size);
            //Whenever we choose a location outside the vector reject the choice and end mutation
            if choice >= self.program.len() {
                break;
            }
            let op = &mut self.program[choice];
            //Randomly mutate only one of the things contained here
            match rng.gen_range(0, 3) {
                0 => (self.mutator)(&mut op.instruction, rng),
                1 => op.first = rng.gen_range(0, choice + self.inputs),
                2 => op.second = rng.gen_range(0, choice + self.inputs),
                _ => unreachable!(),
            }
        }
    }
}

impl<'a, Ins: 'a, R: 'a, Param: 'a, F1: 'a, F2: 'a> SISO<'a, Param, Param> for Mep<Ins, R, Param, F1, F2>
    where F2: Fn(&Ins, Param, Param) -> Param, Param: Clone
{
    type Iter = ResultIterator<'a, Ins, R, Param, F1, F2>;

    fn compute(&'a self, inputs: &'a [Param]) -> Self::Iter {
        //Ensure we have enough opcodes to produce the desired amount of outputs, otherwise the programmer has failed
        assert!(self.outputs <= self.program.len());
        ResultIterator{
            mep: self,
            buff: vec![None; self.program.len()],
            solve_iter: ((self.program.len() + self.inputs - self.outputs)..(self.program.len() + self.inputs)).rev(),
            inputs: inputs,
        }
    }
}

pub struct ResultIterator<'a, Ins: 'a, R: 'a, Param: 'a, F1: 'a, F2: 'a> {
    mep: &'a Mep<Ins, R, Param, F1, F2>,
    buff: Vec<Option<Param>>,
    solve_iter: Rev<Range<usize>>,
    inputs: &'a [Param],
}

impl<'a, Ins, R, Param, F1, F2> ResultIterator<'a, Ins, R, Param, F1, F2>
    where F2: Fn(&Ins, Param, Param) -> Param, Param: Clone
{
    fn op_solved(&mut self, i: usize) -> Param {
        //If this is an input, it is already solved, so return the result immediately
        if i < self.mep.inputs {
            return unsafe{
                self.inputs.get_unchecked(i)
            }.clone();
        }
        //Check if this has been evaluated or not
        let possible = unsafe{
            self.buff.get_unchecked(i - self.mep.inputs)
        }.clone();
        match possible {
            //If it has, return the value
            Some(x) => x,
            //If it hasnt been solved
            None => {
                //Get a reference to the opcode
                let op = unsafe{
                    self.mep.program.get_unchecked(i - self.mep.inputs)
                };
                //Compute the result of the operation, ensuring the inputs are solved beforehand
                let result = (self.mep.processor)(&op.instruction, self.op_solved(op.first), self.op_solved(op.second));
                //Properly store the Some result to the buffer
                unsafe{
                    *self.buff.get_unchecked_mut(i - self.mep.inputs) = Some(result.clone())
                };
                //Return the result
                result
            }
        }
    }
}

impl<'a, Ins, R, Param, F1, F2> Iterator for ResultIterator<'a, Ins, R, Param, F1, F2>
    where F2: Fn(&Ins, Param, Param) -> Param, Param: Clone
{
    type Item = Param;
    fn next(&mut self) -> Option<Param> {
        match self.solve_iter.next() {
            None => None,
            Some(i) => Some(self.op_solved(i)),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Isaac64Rng, SeedableRng, Rng};
    use super::*;
    use super::super::{Genetic, SISO};

    fn mutator(_: &mut i32, _: &mut Isaac64Rng) {}
    fn processor(_: &i32, _: i32, _: i32) -> i32 {0}

    #[test]
    fn new() {
        let a = Mep::new(3, 1, 10, 10, &mut Isaac64Rng::from_seed(&[1, 2, 3, 4]), 0..30, mutator, processor);

        assert_eq!(a.program.iter().map(|i| i.instruction).collect::<Vec<_>>(), (0..30).collect::<Vec<_>>());
    }

    #[test]
    fn crossover() {
        let mut rng = Isaac64Rng::from_seed(&[1, 2, 3, 4]);
        let (a, b) = {
            let mut clos = || Mep::new(3, 1, 10, 10, &mut rng, 0..30, mutator, processor);
            (clos(), clos())
        };
        let old_rngs: Vec<_> = rng.clone().gen_iter::<i32>().take(5).collect();
        let _ = Mep::mate((&a, &b), &mut rng);
        //Ensure that rng was borrowed mutably
        assert!(rng.clone().gen_iter::<i32>().take(5).collect::<Vec<_>>() != old_rngs);
    }

    #[test]
    fn compute() {
        let mut rng = Isaac64Rng::from_seed(&[1, 2, 3, 4]);
        let a = Mep::new(3, 1, 10, 10, &mut rng, 0..30, mutator, processor);

        let inputs = [2, 3, 4];
        assert_eq!(a.compute(&inputs[..]).collect::<Vec<_>>(), [0]);
    }
}
