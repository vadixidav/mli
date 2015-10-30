use std::collections::BTreeSet;

/*
A multi-expression program represented using a series of operations that can reuse results of previous operations.
*/
pub struct Mep<Ins> {
    pub instructions: Vec<Ins>,
}

impl<Ins> Clone for Mep<Ins>
    where Ins: Clone {
    fn clone(&self) -> Self {
        Mep{instructions: self.instructions.clone()}
    }
}

impl<Ins> Mep<Ins> {
    //Generates a new Mep with a particular size and takes a closure to generate random instructions
    pub fn new<F>(total_instructions: usize, mut random_instruction_generator: F) -> Mep<Ins>
        where F: FnMut() -> Ins {
        Mep{instructions: (0..total_instructions).map(|_| random_instruction_generator()).collect()}
    }

    /*
    Performs a crossover that switches at random points in the genome.
    random_point_generator takes an argument that specifies the end of the range to generate and is exclusive.
    */
    pub fn crossover<F>(parent0: &Mep<Ins>, parent1: &Mep<Ins>, points: usize, mut random_point_generator: F) -> Mep<Ins>
        where F: FnMut(usize) -> usize, Ins: Clone {
        //Get the smallest of the two lengths

        let total_instructions = if parent0.instructions.len() < parent1.instructions.len() {
                parent0.instructions.len()
            } else {
                parent1.instructions.len()
            };
        Mep{instructions: (0..points)
            //Generate crossover points
            .map(|_| random_point_generator(total_instructions))
            .chain(Some(total_instructions))
            //Sort them by value into BTree and remove duplicate values
            .fold(BTreeSet::new(), |mut set, i| {set.insert(i); set})
            .iter()
            //Turn every copy of two, prepending a 0, into a range
            .scan(0, |prev, x| {let out = Some(*prev..*x); *prev = *x; out})
            //Enumerate by index to differentiate odd and even values
            .enumerate()
            //Map even pairs to ranges in parent 0 and odd ones to ranges in parent 1 and expand the ranges
            .flat_map(|(index, range)| {
                {if index % 2 == 0 {parent0} else {parent1}}.instructions[range].iter().cloned()
            })
            //Collect all the ranges
            .collect()
        }
    }
}
