# mli

MLI intends to provide modern, data-driven abstractions for machine learning.

MLI provides traits that work much like Combine's `Parser`,
Serde's `Serialize` and `Deserialize`, std's `Iterator`, and future's `Future` and `Stream`.
One should only need to have to create tensor processing primitives and then string them together to craft
a large system that continues to be useful as a primitive in an even larger system.


## Core Crates

- `mli`
    - Core crate with traits
    - Works with `#![no_std]`
- `mli-relu`
    - Contains linear activation functions
    - Doesn't work with `#![no_std]` (blocked by [this](https://github.com/rust-lang/rust/issues/50145))
- `mli-sigmoid`
    - Contains sigmoid activation functions
    - Doesn't work with `#![no_std]` (blocked by [this](https://github.com/rust-lang/rust/issues/50145))
- `mli-conv`
    - Contains convolution implementations
- `mli-ndarray`
    - Allows interoperability between `mli` and `ndarray`
        - Mapping activation functions over tensors

## Goals

- Fast CPU forwards and backwards propogation.
- Abstraction over multiple backends.
- Automatic serialization/deserialization derivations of all state data (using serde).
- Get complete reuse out of tensor code without any overhead for abstractions.
- Allow building tensor pipelines that can talk over streams to allow multi-node setups.
  - It is a non-goal to implement the orchestration of these pipelines in this crate.

## Const Generics

This API is completely usable in its current form, but once const generics are working and stable, this
API will be updated to account for tensor dimensionality at compile time. This likely wont affect the core
`mli` crate, but it will affect the associated types used in several of the other core crates.
