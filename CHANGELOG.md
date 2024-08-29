# CHANGELOG

## [v1.3.0](https://github.com/marcperuz/tilupy/releases/tag/v1.3.0) - 2024-08-29 11:51:45

Minor bug fix + functions to determine the position of the center of mass.

### Feature

- read:
  - add z coordinate to center of mass position ([7e5840c](https://github.com/marcperuz/tilupy/commit/7e5840cfcf149a07ec97519265dc4630e3953fd0)) ([#2](https://github.com/marcperuz/tilupy/pull/2))
  - add function to get center of mass position ([f03777e](https://github.com/marcperuz/tilupy/commit/f03777eb23048fca316bfbde7badc00ce1090a64)) ([#2](https://github.com/marcperuz/tilupy/pull/2))

### Bug Fixes

- make_topo:
  - change cumptrapz to cumulative_trapezoid ([fc9a867](https://github.com/marcperuz/tilupy/commit/fc9a867bd7a96d1908b0a4ab2d78b0f1560444de))

- plot:
  - correct plot_topography function ([4d83e70](https://github.com/marcperuz/tilupy/commit/4d83e70866d0dffb51f644832d3925972b0b701c))
  - correct determination of contour intervals ([2ed4667](https://github.com/marcperuz/tilupy/commit/2ed46672771d11bafddfe2b0c8040696fac6c187))
  - fix contour level determination when ndv is given ([80b78bc](https://github.com/marcperuz/tilupy/commit/80b78bc5ea299790f7f5184e0b1b53df0e24a952))

### Refactor

- read:
  - deal with depreciation warnings ([da92c18](https://github.com/marcperuz/tilupy/commit/da92c183871f0d777637932507beb77188df0f3a))

## [v1.2.0](https://github.com/marcperuz/tilupy/releases/tag/v1.2.0) - 2024-06-07 10:08:07

Function to set an initial mass in Lave2D simulation

### Feature

- lave2D:
  - add function for setting initial mass ([953f07a](https://github.com/marcperuz/tilupy/commit/953f07a76553eca79bf084c9c7478d439c6914ca))

### Documentation

- README:
  - Add references for Lave2D in README ([ed1103c](https://github.com/marcperuz/tilupy/commit/ed1103cb9851340cd49adbb0a08edebf2d68e85c))

## [v1.1.0](https://github.com/marcperuz/tilupy/releases/tag/v1.1.0) - 2024-06-05 16:42:11

Add a module for LAVE2D simulations.
Minor bug corrections and improvements.

### Feature

- lave2D:
  - add read module for lave2D simulations ([8a6f35f](https://github.com/marcperuz/tilupy/commit/8a6f35fb22b291e4854dc15033ba6466af6d07e0)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - add functions to prepare Lave2D simulations ([cb8fcd0](https://github.com/marcperuz/tilupy/commit/cb8fcd0e8f1b5124867eb57c5a8a35efee75cc2f)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - add method for deterimining cell edges from coordinates ([a696a31](https://github.com/marcperuz/tilupy/commit/a696a31ba9cfcd877e9ee50e1ff301d2f516cd54)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - add subpackage for processing LAVE2D ([a69e187](https://github.com/marcperuz/tilupy/commit/a69e1872bc3a4f53312ac1dd0fe08a4d7f6fb271)) ([#1](https://github.com/marcperuz/tilupy/pull/1))

- plot:
  - add automatic shading function ([9c27f91](https://github.com/marcperuz/tilupy/commit/9c27f91021d619d455439fc4fa1a2de78aae165f))
  - change behaviour of saving plots ([9823222](https://github.com/marcperuz/tilupy/commit/98232221f7290693d5451ad9a30b6da305ed5fca))

### Bug Fixes

- lave2D:
  - correct axis coordinates for Modelling domain ([da28b49](https://github.com/marcperuz/tilupy/commit/da28b49f9ab02a3782d244703a2c30b9db5de052)) ([#1](https://github.com/marcperuz/tilupy/pull/1))

- read:
  - get time indexes that must be saved for TemporalResults ([77e83bb](https://github.com/marcperuz/tilupy/commit/77e83bbf589fccb35ebbf968d1d2cdbba8bf5944)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - check existence of folder_out attribute ([6ac28f2](https://github.com/marcperuz/tilupy/commit/6ac28f2e84f8009850e39692b69d0252af50476c)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - fix path merging ([a7ddf7d](https://github.com/marcperuz/tilupy/commit/a7ddf7d11fe16174452e645704d39e07bb6f0787))
  - correct numpy gradient usage ([2b94bef](https://github.com/marcperuz/tilupy/commit/2b94bef0d447945bfa41319f56906367663845e0))

- plot:
  - set colormap when only nan in data ([0b7d818](https://github.com/marcperuz/tilupy/commit/0b7d8180ecc861af0ecab86dcc0dc6e08e05a621)) ([#1](https://github.com/marcperuz/tilupy/pull/1))
  - remove non necessary tight_layout and assertion and  and plot saving options ([e54a82d](https://github.com/marcperuz/tilupy/commit/e54a82df64c92dd92ce2eb08d35358b8514150b2))

- shaltop.read:
  - correct file identifier name ([2f4a0ca](https://github.com/marcperuz/tilupy/commit/2f4a0cafd10c1ea52078b559bca5a9f3d6effd53))
  - ignore blank lines when reading parameters ([328fe03](https://github.com/marcperuz/tilupy/commit/328fe03aa5c95b3607ae6b7e2ee7ef8a967d890f))
  - check for the existence of time_forces.d ([f86d04a](https://github.com/marcperuz/tilupy/commit/f86d04a0343b857344bfab0c3cd7fef896de8017))

- shaltop:
  - correct loading of velocity for shaltop simulations ([175f392](https://github.com/marcperuz/tilupy/commit/175f3925e70410c3d016fb38e3ed3b2ca0a8dd0a))

- gray99:
  - correct setup for gray99 experiment ([bc641b8](https://github.com/marcperuz/tilupy/commit/bc641b88ad7cc516db73176c20f3cbe6bfab1e65))

### Documentation

- README:
  - add installation instruction from conda-forge ([93eccdc](https://github.com/marcperuz/tilupy/commit/93eccdc51b2471abc5104adfc356d9171410fa81))

### Refactor

- plot:
  - change default valueof from_file in Results.read ([b72932e](https://github.com/marcperuz/tilupy/commit/b72932e92820a2ff951963b137874028c8e6135c))

## [v1.0.0](https://github.com/marcperuz/tilupy/releases/tag/v1.0.0) - 2024-02-01 13:40:53

Management of spatial and temporal integration of simulation results. Major improvements include 
 -  Automatic and intelligent management of notations (symbols, litteral names, units)
 - Automatic generation of notations when results are processed (e.g. time or spatial integrals)
 - Generic plot functions for various result types, in particular time dependent results (line plots and "shotgather" type plots)
 - Simplified and intuitive recovery of simulations results from strings, including reading code-specific outputs

## [v0.1.5](https://github.com/marcperuz/tilupy/releases/tag/v0.1.5) - 2024-01-04 12:54:39

New release with minor bugs/style corrections

### Feature

- data:
  - add Gray99 topography and mass as data ([b02e092](https://github.com/marcperuz/tilupy/commit/b02e09276e15ffcdefc486b497b8f64d64021da5))

- make_mass:
  - add module for initial synthetic mass generation ([ca39f1b](https://github.com/marcperuz/tilupy/commit/ca39f1b26d6c69624eb568180f67fe89acf0cb22))

- plot:
  - add automatic calculation of contour line intervals ([934b1ce](https://github.com/marcperuz/tilupy/commit/934b1ce5f47f30956d2572993ec4e587f1bc519b))

- make_topo:
  - add function to generate synthetic channels. ([e8d5de0](https://github.com/marcperuz/tilupy/commit/e8d5de0b05cdcde1c30f54e0dff852329395e64c))
  - generate topography from the experiment of Gray99 ([6005075](https://github.com/marcperuz/tilupy/commit/6005075c17ae72ff53833a60ca2e5ed11918e76d))

### Bug Fixes

- tests:
  - fix indentation error in shaltop test function ([53c9249](https://github.com/marcperuz/tilupy/commit/53c92493e2efa67d379675565ce53e4aee9d1971))

- calibration:
  - fix old version of module import not corrected ([1af2a4a](https://github.com/marcperuz/tilupy/commit/1af2a4aec3c63e0663c45fb349d8b423d2cee2b6))

- test:
  - change test assertion in test_generated_data.py ([b132750](https://github.com/marcperuz/tilupy/commit/b13275075b6908fb5820879fc4daca857f6e2ad8))

### Documentation

- README:
  - correct typo ([9985c8f](https://github.com/marcperuz/tilupy/commit/9985c8f9559b67b6da7b86231601753e9f7755cc))

## [v0.1.4](https://github.com/marcperuz/tilupy/releases/tag/v0.1.4) - 2023-07-20 14:17:06

Compatibility with python 3.8 to 3.11 is ensured and tested.

### Bug Fixes

- test_shaltop:
  - replace os.rmdir by shutil.rmtree ([2c8c7e6](https://github.com/marcperuz/tilupy/commit/2c8c7e6e7b86247d1016b48597da354592c66ba6))

- tilupy.cmd:
  - Use old behaviour of glob.glob, without root_dir parameter ([223dedb](https://github.com/marcperuz/tilupy/commit/223dedb4dd0996e8eee42a61c6ad6f50b6171b47))

### Documentation

- version:
  - increase version to 0.1.4 ([55ed352](https://github.com/marcperuz/tilupy/commit/55ed3522ac8b1d080114630e02faf2f341d89866))

## [v0.1.3](https://github.com/marcperuz/tilupy/releases/tag/v0.1.3) - 2023-07-20 12:52:45

Minimum python required is >=3.10, needed for glob module

## [v0.1.2](https://github.com/marcperuz/tilupy/releases/tag/v0.1.2) - 2023-07-20 12:33:17

Realease first version to PyPi

### Documentation

- version:
  - change version for testing pip install in test-pypi ([8e8398c](https://github.com/marcperuz/tilupy/commit/8e8398c755cfb946470cc36d5e3f94ef777d5802))
  - Change version to test download to test-pypi ([97c9013](https://github.com/marcperuz/tilupy/commit/97c9013581fb6788efdf9272538ae89a5de3bd73))

## [v0.1.0](https://github.com/marcperuz/tilupy/releases/tag/v0.1.0) - 2023-07-20 09:01:35

*No description*

### Documentation

- pyproject.toml:
  - <subject>Add correct license classifier ([9687e59](https://github.com/marcperuz/tilupy/commit/9687e590dea3b9ca0f0fb7ea9b4a4d52bbbefd28))

\* *This CHANGELOG was automatically generated by [auto-generate-changelog](https://github.com/BobAnkh/auto-generate-changelog)*
