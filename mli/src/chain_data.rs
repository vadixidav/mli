use core::iter::Chain;

/// `ChainData` derives several traits and attempts to allow chaining the IntoIterator
/// implementation over `&` and `&mut` to allow access to the data within.
/// This is critical to allow optimizers to perform per-weight gradient update rules.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChainData<A, B>(pub A, pub B);

impl<'a, A, B> IntoIterator for &'a ChainData<A, B>
    where &'a A: IntoIterator<Item=<&'a B as IntoIterator>::Item>, &'a B: IntoIterator
{
    type IntoIter = Chain<<&'a A as IntoIterator>::IntoIter, <&'a B as IntoIterator>::IntoIter>;
    type Item = <&'a A as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).into_iter().chain((&self.1).into_iter())
    }
}

impl<'a, A, B> IntoIterator for &'a mut ChainData<A, B>
    where &'a mut A: IntoIterator<Item=<&'a mut B as IntoIterator>::Item>, &'a mut B: IntoIterator
{
    type IntoIter = Chain<<&'a mut A as IntoIterator>::IntoIter, <&'a mut B as IntoIterator>::IntoIter>;
    type Item = <&'a mut A as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.0).into_iter().chain((&mut self.1).into_iter())
    }
}