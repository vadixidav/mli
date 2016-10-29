extern crate num;
use std::collections::BTreeSet;
use std::cmp;
use rand::{Rng, Rand};
use std::ops::Range;
use std::iter::Rev;
use super::{Stateless, MateRand, Mutate};

/// Defines an opcode for the Mep. Every opcode contains an instruction and two parameter indices.
/// These specify which previous opcodes produced the result required as inputs to this opcode.
/// These parameters can also come from the inputs to the program, which sequentially
/// proceed the internal instructions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Opcode<Ins> {
    instruction: Ins,
    first: usize,
    second: usize,
}

/// A multi-expression program represented using a series of operations that can reuse
/// results of previous operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mep<Ins> {
    program: Vec<Opcode<Ins>>,
    mutate_lambda: usize,
    crossover_points: usize,
    inputs: usize,
    outputs: usize,
}

impl<Ins> Mep<Ins> {
    /// Generates a new Mep with a particular size and takes a closure to generate random
    /// instructions. Takes an RNG as well to generate random internal data for each instruction.
    ///
    /// `mutate_lambda` corresponds to the lambda of a poisson distribution. It is proportional
    /// to the average unit mutate cycles between mutations. It is biased by 1, so a `mutate_lambda`
    /// of 0 means it will be mutated every cycle.
    ///
    /// `crossover_points` corresponds to the maximum amount of crossover locations on the
    /// chromosome when mating. When mating the `crossover_points` is chosen randomly between the
    /// two Meps.
    pub fn new<R>(inputs: usize,
                  outputs: usize,
                  internal_instruction_count: usize,
                  mutate_lambda: usize,
                  crossover_points: usize,
                  rng: &mut R)
                  -> Self
        where R: Rng,
              Ins: Rand
    {
        Mep {
            program: (0..internal_instruction_count + outputs)
                .map(|i| if i < internal_instruction_count {
                    Opcode {
                        instruction: rng.gen(),
                        first: rng.gen_range(0, i + inputs),
                        second: rng.gen_range(0, i + inputs),
                    }
                } else {
                    Opcode {
                        instruction: rng.gen(),
                        first: rng.gen_range(0, internal_instruction_count + inputs),
                        second: rng.gen_range(0, internal_instruction_count + inputs),
                    }
                })
                .collect(),
            mutate_lambda: mutate_lambda,
            crossover_points: crossover_points,
            inputs: inputs,
            outputs: outputs,
        }
    }
}

impl<Ins, R> MateRand<R> for Mep<Ins>
    where R: Rng,
          Ins: Clone
{
    fn mate(&self, rhs: &Self, rng: &mut R) -> Self {
        // Each Mep must have the same amount of inputs
        // TODO: Once Rust implements generic values, this can be made explicit and is not needed
        assert!(self.inputs == rhs.inputs);
        assert!(self.outputs == rhs.outputs);
        // Get the smallest of the two lengths
        let total_instructions = cmp::min(self.program.len(), rhs.program.len());
        let crossover_choice = rng.gen_range(0, 2);
        Mep {
            program:
                // Generate a randomly sized sequence between 1 and the minimum between
                // `crossover_points` vs `total_instructions / 2`.
                (0..rng.gen_range(1, cmp::min(total_instructions / 2, {
                if crossover_choice == 0 {self}
                    else {rhs}}.crossover_points + 1)))
                // Map these to random crossover points.
                .map(|_| rng.gen_range(0, total_instructions))
                // Add total_instructions at the end so we can generate a range with it.
                .chain(Some(total_instructions))
                // Sort them by value into BTree, which removes duplicates and orders them.
                .fold(BTreeSet::new(), |mut set, i| {set.insert(i); set})
                // Iterate over the sorted values.
                .iter()
                // Turn every copy of two, prepending a 0, into a range.
                .scan(0, |prev, x| {let out = Some(*prev..*x); *prev = *x; out})
                // Enumerate by index to differentiate odd and even values.
                .enumerate()
                // Map even pairs to ranges in parent 0 and odd ones to ranges in
                // parent 1 and expand the ranges.
                .flat_map(|(index, range)| {
                    {if index % 2 == 0 {self} else {rhs}}.program[range].iter().cloned()
                })
                // Collect all the instruction ranges from each parent.
                .collect(),

            mutate_lambda: if self.mutate_lambda < rhs.mutate_lambda {
                rng.gen_range(self.mutate_lambda, rhs.mutate_lambda + 1)
            } else {
                rng.gen_range(rhs.mutate_lambda, self.mutate_lambda + 1)
            },

            crossover_points: if self.crossover_points < rhs.crossover_points {
                rng.gen_range(self.crossover_points, rhs.crossover_points + 1)
            } else {
                rng.gen_range(rhs.crossover_points, self.crossover_points + 1)
            },

            inputs: self.inputs,
            outputs: self.outputs,
        }
    }
}

impl<Ins, R> Mutate<R> for Mep<Ins>
    where Ins: Mutate<R>,
          R: Rng
{
    fn mutate(&mut self, rng: &mut R) {
        // For this entire cycle, the biased lambda from the previous cycle is effective.
        let effective_lambda = self.mutate_lambda + 1;

        // Mutate `mutate_lambda`.
        if rng.gen_range(0, effective_lambda) == 0 {
            // Make it possibly go up or down by 1.
            match rng.gen_range(0, 2) {
                0 => self.mutate_lambda = self.mutate_lambda.saturating_add(1),
                _ => self.mutate_lambda = self.mutate_lambda.saturating_sub(1),
            }
        }

        // Mutate `crossover_points`.
        if rng.gen_range(0, effective_lambda) == 0 {
            // Make it possibly go up or down by 1.
            match rng.gen_range(0, 2) {
                0 => self.crossover_points = self.crossover_points.saturating_add(1),
                _ => self.crossover_points = self.crossover_points.saturating_sub(1),
            }
        }

        // Get the program length.
        let plen = self.program.len();

        // Mutate the instructions.
        loop {
            // Choose a random location in the instructions and then add a random value.
            // up to the unit_mutate_size.
            let choice = rng.gen_range(0, plen + effective_lambda);
            // Whenever we choose a location outside the vector reject the choice and end mutation.
            if choice >= plen {
                break;
            }
            let op = &mut self.program[choice];
            // Randomly mutate only one of the things contained here.
            match rng.gen_range(0, 3) {
                0 => op.instruction.mutate(rng),
                1 => op.first = if choice > plen  - self.outputs {
                    // Handle the case where an output is selected.
                    rng.gen_range(0, choice + plen  - self.outputs)
                } else {
                    rng.gen_range(0, choice + self.inputs)
                },
                _ => op.second = if choice > plen  - self.outputs {
                    // Handle the case where an output is selected.
                    rng.gen_range(0, choice + plen  - self.outputs)
                } else {
                    rng.gen_range(0, choice + self.inputs)
                },
            }
        }
    }
}

impl<'a, Ins, Param> Stateless<'a, &'a [Param], ResultIterator<'a, Ins, Param>> for Mep<Ins>
    where Param: Clone
{
    fn process(&'a self, inputs: &'a [Param]) -> ResultIterator<'a, Ins, Param> {
        ResultIterator {
            mep: self,
            buff: vec![None; self.program.len()],
            solve_iter: ((self.program.len() + self.inputs - self.outputs)..(self.program.len() +
                                                                             self.inputs))
                .rev(),
            inputs: inputs,
        }
    }
}

pub struct ResultIterator<'a, Ins, Param>
    where Ins: 'a,
          Param: 'a
{
    mep: &'a Mep<Ins>,
    buff: Vec<Option<Param>>,
    solve_iter: Rev<Range<usize>>,
    inputs: &'a [Param],
}

impl<'a, Ins, Param> ResultIterator<'a, Ins, Param> {
    #[inline]
    fn op_solved(&mut self, i: usize) -> Param
        where Param: Clone,
              Ins: Stateless<'a, (Param, Param), Param>
    {
        // If this is an input, it is already solved, so return the result immediately.
        if i < self.mep.inputs {
            return unsafe { self.inputs.get_unchecked(i) }.clone();
        }
        // Check if this has been evaluated or not.
        let possible = unsafe { self.buff.get_unchecked(i - self.mep.inputs) }.clone();
        match possible {
            // If it has, return the value.
            Some(x) => x,
            // If it hasn't been solved.
            None => {
                // Get a reference to the opcode.
                let op = unsafe { self.mep.program.get_unchecked(i - self.mep.inputs) };
                // Compute the result of the operation, ensuring the inputs are solved beforehand.
                let result = op.instruction
                    .process((self.op_solved(op.first), self.op_solved(op.second)));
                // Properly store the Some result to the buffer.
                unsafe { *self.buff.get_unchecked_mut(i - self.mep.inputs) = Some(result.clone()) };
                // Return the result.
                result
            }
        }
    }
}

impl<'a, Ins, Param> Iterator for ResultIterator<'a, Ins, Param>
    where Param: Clone,
          Ins: Stateless<'a, (Param, Param), Param>
{
    type Item = Param;
    #[inline]
    fn next(&mut self) -> Option<Param> {
        match self.solve_iter.next() {
            None => None,
            Some(i) => Some(self.op_solved(i)),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Isaac64Rng, SeedableRng, Rand};
    use super::*;
    use super::super::{MateRand, Mutate, Stateless};

    #[derive(Clone)]
    struct Op;

    impl<R> Mutate<R> for Op {
        fn mutate(&mut self, _: &mut R) {}
    }

    impl<'a> Stateless<'a, (i32, i32), i32> for Op {
        fn process(&'a self, inputs: (i32, i32)) -> i32 {
            inputs.0 + inputs.1
        }
    }

    impl Rand for Op {
        fn rand<R>(_: &mut R) -> Self {
            Op
        }
    }

    #[test]
    fn mep() {
        let mut rng = Isaac64Rng::from_seed(&[1, 2, 3, 4]);
        let (mut a, b) = {
            let mut clos = || Mep::<Op>::new(3, 1, 10, 10, 10, &mut rng);
            (clos(), clos())
        };
        a.mutate(&mut rng);
        let c = a.mate(&b, &mut rng);

        let inputs = [2i32, 3, 4];
        c.process(&inputs[..]).collect::<Vec<_>>();
    }
}
