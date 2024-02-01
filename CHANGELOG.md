# CHANGELOG

## [v1.0.0-rc1](https://github.com/marcperuz/tilupy/releases/tag/v1.0.0-rc1) - 2024-02-01 13:20:38

pre-release for 1.0.0, checking that distribution to test-pypi is working

### Feature

- plot:
  - fix and improve minor aspects of plot ([f393926](https://github.com/marcperuz/tilupy/commit/f3939267f11d3a80235cbbfc3430bcb6edee601a))
  - display legend for line time plots. ([8e4a5b4](https://github.com/marcperuz/tilupy/commit/8e4a5b4ae2ba809003d5a9fed56d7802e41ebd41))
  - add default legend to colobar in shotgather plots ([d5a1043](https://github.com/marcperuz/tilupy/commit/d5a104352cdfb92245f9e5946032cf532380f8ad))
  - change scale to auto for shotgather plots ([56504ae](https://github.com/marcperuz/tilupy/commit/56504ae731443d4a783d293c46b4ce0b74fb35f9))

- read:
  - add custom notation to forces for shaltop results ([f99b8a4](https://github.com/marcperuz/tilupy/commit/f99b8a454c3cb822e0dc6ff0b682dac2b5ff484b))
  - generalize reading and processing of simulation outputs ([d533e23](https://github.com/marcperuz/tilupy/commit/d533e23d509d338247588a0a48283b1f6f62833b))
  - improve processing of Temporalresults ([70957f2](https://github.com/marcperuz/tilupy/commit/70957f2f4dfe3c54678f1dbf67ed58db3a581916))
  - add function to process spatially TemporalOutput ([c826fb2](https://github.com/marcperuz/tilupy/commit/c826fb24dde2f60cf1d863910088481c6c5e018b))
  - add spatial integration function to get_temporal_stat ([8ef1096](https://github.com/marcperuz/tilupy/commit/8ef1096d1a1ab9662598ddb38137645ade33fd89))

- notation:
  - change symbol for integral operator ([72754a1](https://github.com/marcperuz/tilupy/commit/72754a125fbb5c539b3a6cf068b14625a0d5a725))

- notations:
  - improve management of notations ([079ab1a](https://github.com/marcperuz/tilupy/commit/079ab1ab73682b350a5c7b0c69cfe9d503f308af))

- TemporalOutput1D:
  - add coords as parameter of plot method ([2c5b334](https://github.com/marcperuz/tilupy/commit/2c5b3341882122cab5689032aca54a69bf426a3d))

- data:
  - add names of processed outputs ([0607c55](https://github.com/marcperuz/tilupy/commit/0607c55fbb930ec702caf042c9a8d3414cac9fdc))

- shaltop:
  - add function to generate slurm job/conf files ([031a5a9](https://github.com/marcperuz/tilupy/commit/031a5a9af9d3e7b9f92af81b9bb66e826a05ba98))

### Bug Fixes

- read.shaltop:
  - correct cases when to read_from_file ([292e643](https://github.com/marcperuz/tilupy/commit/292e643f4eba8909faf4380da62305d52211937f))

- read:
  - fix orientation issue in get_spatial_stat ([b1213ee](https://github.com/marcperuz/tilupy/commit/b1213eee0d77e827c8a4a7839ea9f7752a1cc82a))
  - fix spatial integration axis to match x/y direction ([b7fd4ea](https://github.com/marcperuz/tilupy/commit/b7fd4ea0ea3c617bcae3c03bba9b3f527d69ca84))
  - change time intagral name ([b7cc0ff](https://github.com/marcperuz/tilupy/commit/b7cc0ff1a976be19863bc83603c11b098094be75))
  - fix time integral computation ([2dc6040](https://github.com/marcperuz/tilupy/commit/2dc6040f9ec984ff80a34b9906207006afcbce06))

- plot:
  - Fix shotgather plot ([b5e31ba](https://github.com/marcperuz/tilupy/commit/b5e31badc846cf82e6d98eb8166a1f6f9855bdc0))

### Refactor

- plot:
  - change function for getting colormap ([9b50e72](https://github.com/marcperuz/tilupy/commit/9b50e7239e5a69540ec476c1ab615939d36144d1))
  - add separate function for sutomized imshow ([db9fabc](https://github.com/marcperuz/tilupy/commit/db9fabcb41e30f560c3aeffee5dea728a769b241))

## [v0.1.5.rc2](https://github.com/marcperuz/tilupy/releases/tag/v0.1.5.rc2) - 2024-01-11 16:37:08

*No description*

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
