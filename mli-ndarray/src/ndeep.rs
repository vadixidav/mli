use ndarray::{ArrayBase, Data, DataClone, DataMut, Dimension};
use std::fmt::{self, Debug};
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// This struct adapts ndarray's `ArrayBase`, regardless of its form, to act entirely element-wise
/// for all operations. This is important because [`mli::Backward::TrainDelta`] needs to support
/// element-wise operations for different optimizers other than standard gradient descent to work.
pub struct Ndeep<S: Data, D>(pub ArrayBase<S, D>);

impl<S, D: Dimension + Clone> Clone for Ndeep<S, D>
where
    S: DataClone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<S, D: Dimension + Clone> Debug for Ndeep<S, D>
where
    S: Data,
    <S as Data>::Elem: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<S, D: Dimension> AddAssign<f32> for Ndeep<S, D>
where
    S: DataMut<Elem = f32>,
{
    fn add_assign(&mut self, rhs: S::Elem) {
        for n in self.0.iter_mut() {
            *n += rhs;
        }
    }
}

impl<S, D: Dimension> AddAssign<f64> for Ndeep<S, D>
where
    S: DataMut<Elem = f64>,
{
    fn add_assign(&mut self, rhs: S::Elem) {
        for n in self.0.iter_mut() {
            *n += rhs;
        }
    }
}

impl<S, D: Dimension> AddAssign for Ndeep<S, D>
where
    S: DataMut,
    S::Elem: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        for (n, &r) in self.0.iter_mut().zip(rhs.0.iter()) {
            *n += r;
        }
    }
}

impl<S, D: Dimension> MulAssign<f32> for Ndeep<S, D>
where
    S: DataMut<Elem = f32>,
{
    fn mul_assign(&mut self, rhs: S::Elem) {
        for n in self.0.iter_mut() {
            *n *= rhs;
        }
    }
}

impl<S, D: Dimension> MulAssign<f64> for Ndeep<S, D>
where
    S: DataMut<Elem = f64>,
{
    fn mul_assign(&mut self, rhs: S::Elem) {
        for n in self.0.iter_mut() {
            *n *= rhs;
        }
    }
}

impl<S, D: Dimension> MulAssign for Ndeep<S, D>
where
    S: DataMut,
    S::Elem: MulAssign + Copy,
{
    fn mul_assign(&mut self, rhs: Self) {
        for (n, &r) in self.0.iter_mut().zip(rhs.0.iter()) {
            *n *= r;
        }
    }
}

impl<S, D: Dimension> Add<S::Elem> for Ndeep<S, D>
where
    S: DataMut,
    S::Elem: AddAssign + Copy,
{
    type Output = Self;

    fn add(mut self, rhs: S::Elem) -> Self {
        for n in self.0.iter_mut() {
            *n += rhs;
        }
        self
    }
}

impl<S, D: Dimension> Mul<S::Elem> for Ndeep<S, D>
where
    S: DataMut,
    S::Elem: MulAssign + Copy,
{
    type Output = Self;

    fn mul(mut self, rhs: S::Elem) -> Self {
        for n in self.0.iter_mut() {
            *n *= rhs;
        }
        self
    }
}
