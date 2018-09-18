//! MLI intends to provide modern, data-driven abstractions for machine learning.
//!
//! MLI provides traits that work much like Combine's `Parser`,
//! Serde's `Serialize` and `Deserialize`, std's `Iterator`, and future's `Future` and `Stream`.
//! One should only need to have to create tensor processing primitives and then string them together to craft
//! a large system that continues to be useful as a primitive in an even larger system.
//!
//! ## Goals
//! - Very fast CPU forwards and backwards propogation.
//! - Abstraction over multiple backends.
//! - Automatic serialization/deserialization  derivations of all state data (using serde).
//! - Get complete reuse out of tensor code without any overhead for abstractions.
//! - Allow building tensor pipelines that can talk over streams to allow multi-node setups.
//!   - It is a non-goal to implement the orchestration of these pipelines in this crate.
//!
//! ## Const Generics
//! Const generics must land before this API can be stabilized. This API will not be able to interact
//! with tensor sizes and dimensions to allow static checking without const generics. Until that lands in stable
//! and nalgebra is updated to use it, this library will continue to be in an unstable state.

#![no_std]
