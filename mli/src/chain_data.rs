use core::ops::{Add, AddAssign, Mul, MulAssign};
use num_traits::{One, Zero};

/// `ChainData` derives several traits and attempts to allow chaining the IntoIterator
/// implementation over `&` and `&mut` to allow access to the data within.
/// This is critical to allow optimizers to perform per-weight gradient update rules.
#[derive(Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChainData<A, B>(pub A, pub B);

impl<A, B> Add for ChainData<A, B>
where
    A: Add,
    B: Add,
{
    type Output = ChainData<A::Output, B::Output>;

    fn add(self, rhs: Self) -> Self::Output {
        ChainData(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl<A, B> Add<f32> for ChainData<A, B>
where
    A: Add<f32>,
    B: Add<f32>,
{
    type Output = ChainData<A::Output, B::Output>;

    fn add(self, rhs: f32) -> Self::Output {
        ChainData(self.0 + rhs, self.1 + rhs)
    }
}

impl<A, B> Add<f64> for ChainData<A, B>
where
    A: Add<f64>,
    B: Add<f64>,
{
    type Output = ChainData<A::Output, B::Output>;

    fn add(self, rhs: f64) -> Self::Output {
        ChainData(self.0 + rhs, self.1 + rhs)
    }
}

impl<A, B> AddAssign for ChainData<A, B>
where
    A: AddAssign,
    B: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl<A, B> AddAssign<f32> for ChainData<A, B>
where
    A: AddAssign<f32>,
    B: AddAssign<f32>,
{
    fn add_assign(&mut self, rhs: f32) {
        self.0 += rhs;
        self.1 += rhs;
    }
}

impl<A, B> AddAssign<f64> for ChainData<A, B>
where
    A: AddAssign<f64>,
    B: AddAssign<f64>,
{
    fn add_assign(&mut self, rhs: f64) {
        self.0 += rhs;
        self.1 += rhs;
    }
}

impl<A, B> Zero for ChainData<A, B>
where
    A: Zero + Add,
    B: Zero + Add,
{
    fn zero() -> Self {
        Self(A::zero(), B::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

impl<A, B> Mul for ChainData<A, B>
where
    A: Mul,
    B: Mul,
{
    type Output = ChainData<A::Output, B::Output>;

    fn mul(self, rhs: Self) -> Self::Output {
        ChainData(self.0 * rhs.0, self.1 * rhs.1)
    }
}

impl<A, B> MulAssign for ChainData<A, B>
where
    A: MulAssign,
    B: MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
        self.1 *= rhs.1;
    }
}

impl<A, B> MulAssign<f32> for ChainData<A, B>
where
    A: MulAssign<f32>,
    B: MulAssign<f32>,
{
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
        self.1 *= rhs;
    }
}

impl<A, B> MulAssign<f64> for ChainData<A, B>
where
    A: MulAssign<f64>,
    B: MulAssign<f64>,
{
    fn mul_assign(&mut self, rhs: f64) {
        self.0 *= rhs;
        self.1 *= rhs;
    }
}

impl<A, B> One for ChainData<A, B>
where
    A: One + Mul,
    B: One + Mul,
{
    fn one() -> Self {
        Self(A::one(), B::one())
    }
}
